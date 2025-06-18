import os
import configparser

import requests
from functools import wraps
import jwt
from jwt import PyJWKClient

from datetime import datetime
from flask import Flask, request, jsonify, json, Response, abort
from flask_cors import CORS
from hana_ml import dataframe
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sql_formatter.core import format_sql
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

# Check if the application is running on Cloud Foundry
if 'VCAP_APPLICATION' in os.environ:
    # Running on Cloud Foundry, use environment variables
    from app.utilities_hana import kmeans_and_tsne  # works in CF
    
    hanaURL = os.getenv('DB_ADDRESS')
    hanaPort = os.getenv('DB_PORT')
    hanaUser = os.getenv('DB_USER')
    hanaPW = os.getenv('DB_PASSWORD')
    
    XSUAA_URL = os.getenv("XSUAA_URL")  # e.g. https://<subdomain>.authentication.<region>.hana.ondemand.com
    XSUAA_CLIENT_ID = os.getenv("XSUAA_CLIENT_ID")
    XSUAA_CLIENT_SECRET = os.getenv("XSUAA_CLIENT_SECRET")
    XSUAA_XSAPPNAME = os.getenv("XSUAA_XSAPPNAME")

else:    
    # Not running on Cloud Foundry, read from config.ini file
    from utilities_hana import kmeans_and_tsne  # works in local machine
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    hanaURL = config['database']['address']
    hanaPort = config['database']['port']
    hanaUser = config['database']['user']
    hanaPW = config['database']['password']

    XSUAA_URL = config['xsuaa']['url']
    XSUAA_CLIENT_ID = config['xsuaa']['clientid']
    XSUAA_CLIENT_SECRET = config['xsuaa']['clientsecret']
    XSUAA_XSAPPNAME = config['xsuaa']['appname']

def get_xsuaa_public_key():
    # Get the public key from XSUAA's JWKS endpoint
    resp = requests.get(f"{XSUAA_URL}/token_keys")
    jwks = resp.json()
    return {key['kid']: key for key in jwks['keys']}

def require_oauth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", None)
        if not auth or not auth.startswith("Bearer "):
            abort(401, description="Missing or invalid Authorization header")
        token = auth.split(" ")[1]
        try:
            jwks_url = f"{XSUAA_URL}/token_keys"
            jwks_client = PyJWKClient(jwks_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            public_key = signing_key.key
            decoded = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=XSUAA_CLIENT_ID,
                options={"verify_exp": True}
            )
            # Optionally: check scopes, xsappname, etc.
        except Exception as e:
            abort(401, description=f"Invalid token: {str(e)}")
        return f(*args, **kwargs)
    return decorated

# Establish a connection to SAP HANA
connection = dataframe.ConnectionContext(hanaURL, hanaPort, hanaUser, hanaPW)

# Initialize the proxy client and LLM model globally
proxy_client = get_proxy_client('gen-ai-hub')
llm = ChatOpenAI(proxy_model_name='gpt-4', temperature=0)

app = Flask(__name__)
CORS(app)

# Function to create the CATEGORIES table if it doesn't exist
def create_categories_table_if_not_exists():
    create_table_sql = """
        DO BEGIN
            DECLARE table_exists INT;
            
            -- Check and create CATEGORIES table
            SELECT COUNT(*) INTO table_exists
            FROM SYS.TABLES 
            WHERE TABLE_NAME = 'CATEGORIES' AND SCHEMA_NAME = CURRENT_SCHEMA;
            
            IF table_exists = 0 THEN
                CREATE TABLE CATEGORIES (
                    "index" INTEGER,
                    "category_label" NVARCHAR(100),
                    "category_descr" NVARCHAR(5000),
                    "category_embedding" REAL_VECTOR 
                        GENERATED ALWAYS AS VECTOR_EMBEDDING("category_descr", 'DOCUMENT', 'SAP_NEB.20240715')
                );
            END IF;
        END;
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(create_table_sql)
    cursor.close()
    
# Function to create the PROJECT_BY_CATEGORY table if it doesn't exist
def create_project_by_category_table_if_not_exists():
    create_table_sql = """
        DO BEGIN
            DECLARE table_exists INT;
            
            -- Check and create PROJECT_BY_CATEGORY table
            SELECT COUNT(*) INTO table_exists
            FROM SYS.TABLES 
            WHERE TABLE_NAME = 'PROJECT_BY_CATEGORY' AND SCHEMA_NAME = CURRENT_SCHEMA;
            
            IF table_exists = 0 THEN
                CREATE TABLE PROJECT_BY_CATEGORY (
                    PROJECT_ID INT,
                    CATEGORY_ID INT
                );
            END IF;
        END;
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(create_table_sql)
    cursor.close()  
    
@app.route('/update_categories_and_projects', methods=['POST'])
@require_oauth
def update_categories_and_projects():
    data = request.get_json()
    categories = data
    
    if not categories:
        return jsonify({"error": "No categories provided"}), 400
    
    cursor = connection.connection.cursor()
    
    # Ensure the CATEGORIES table exists
    create_categories_table_if_not_exists()
    
    # Drop existing values from the CATEGORIES table
    cursor.execute("TRUNCATE TABLE CATEGORIES")
    
    # Ensure the PROJECT_BY_CATEGORY table exists
    create_project_by_category_table_if_not_exists()
    
    # Drop existing values from the PROJECT_BY_CATEGORY table
    cursor.execute("TRUNCATE TABLE PROJECT_BY_CATEGORY")
    
    # Add custom categories to the CATEGORIES table
    for index, (title, description) in enumerate(categories.items()):
        insert_sql = f"""
            INSERT INTO CATEGORIES ("index", "category_label", "category_descr")
            VALUES ({index}, '{title.replace("'", "''")}', '{description.replace("'", "''")}')
        """
        cursor.execute(insert_sql)
    
    # Retrieve categories from the CATEGORIES table
    categories_df = dataframe.DataFrame(connection, 'SELECT * FROM CATEGORIES')
    
    # Retrieve topics from the ADVISORIES table
    advisories_df = dataframe.DataFrame(connection, 'SELECT "project_number", "topic" FROM ADVISORIES4')
    
    # Iterate over each advisory and calculate the most similar category
    for advisory in advisories_df.collect().to_dict(orient='records'):
        # print("Advisory columns:", advisory.keys())
        project_number = advisory['project_number']
        topic = advisory['topic']
        
        # Check if project_number is an integer
        if not isinstance(project_number, int):
            print(f"Skipping project_number={project_number} as it is not an integer")
            continue
    
        similarities = []
        # Iterate over each category and calculate the similarity
        for category in categories_df.collect().to_dict(orient='records'):

            category_id = category['index']
            category_description = category['category_descr']
            
            # Use HANA SQL for COSINE similarity
            similarity_sql = f"""
                SELECT COSINE_SIMILARITY(
                    VECTOR_EMBEDDING('{topic.replace("'", "''")}', 'DOCUMENT', 'SAP_NEB.20240715'),
                    VECTOR_EMBEDDING('{category_description.replace("'", "''")}', 'DOCUMENT', 'SAP_NEB.20240715')
                ) AS similarity
                FROM DUMMY
            """

            similarity_df = dataframe.DataFrame(connection, similarity_sql)
            similarity_results = similarity_df.collect()
            
            if not similarity_results.empty:
                similarity = similarity_results.iloc[0]['SIMILARITY']
                similarities.append((category_id, similarity))
            else:
                print(f"No similarity result for category_id={category_id} and topic={topic}")

        # Find the most similar category
        if similarities:
            most_similar_category = max(similarities, key=lambda x: x[1])
            category_id = most_similar_category[0]

            # Update PROJECT_BY_CATEGORY table
            insert_sql = f"""
                INSERT INTO "PROJECT_BY_CATEGORY" ("PROJECT_ID", "CATEGORY_ID")
                VALUES ('{project_number}', {category_id})
            """
            cursor.execute(insert_sql)
        else:
            print(f"No valid similarities found for project_number={project_number}")
    
    cursor.close()
    return jsonify({"message": "Categories and project categories updated successfully"}), 200

@app.route('/get_all_project_categories', methods=['GET'])
@require_oauth
def get_all_project_categories():
    # SQL query to retrieve all records from the PROJECT_BY_CATEGORY table
    sql_query = """
        SELECT pbc."PROJECT_ID", c."category_label"
        FROM "PROJECT_BY_CATEGORY" pbc
        JOIN "CATEGORIES" c ON pbc."CATEGORY_ID" = c."index"
    """
    hana_df = dataframe.DataFrame(connection, sql_query)
    project_categories = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = project_categories.to_dict(orient='records')
    return jsonify({"project_categories": results}), 200

@app.route('/get_categories', methods=['GET'])
def get_categories():
    # SQL query to retrieve all records from the CATEGORIES table
    sql_query = """
        SELECT "index", "category_label", "category_descr"
        FROM "CATEGORIES"
    """
    hana_df = dataframe.DataFrame(connection, sql_query)
    categories = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = categories.to_dict(orient='records')
    return jsonify(results), 200

# Function to create the CLUSTERING table if it doesn't exist
def create_clustering_table_if_not_exists():
    create_table_sql = """
        DO BEGIN
            DECLARE table_exists INT;
            SELECT COUNT(*) INTO table_exists
            FROM SYS.TABLES 
            WHERE TABLE_NAME = 'CLUSTERING' AND SCHEMA_NAME = CURRENT_SCHEMA;
            
            IF table_exists = 0 THEN
                CREATE TABLE CLUSTERING (
                    PROJECT_NUMBER NVARCHAR(255),
                    x DOUBLE,
                    y DOUBLE,
                    CLUSTER_ID INT
                );
            END IF;
            
            -- Check and create CLUSTERING_DATA table
            SELECT COUNT(*) INTO table_exists
            FROM SYS.TABLES 
            WHERE TABLE_NAME = 'CLUSTERING_DATA' AND SCHEMA_NAME = CURRENT_SCHEMA;
            
            IF table_exists = 0 THEN
                CREATE TABLE CLUSTERING_DATA (
                    CLUSTER_ID INT,
                    CLUSTER_DESCRIPTION NVARCHAR(255),
                    EMBEDDING REAL_VECTOR GENERATED ALWAYS AS VECTOR_EMBEDDING(CLUSTER_DESCRIPTION, 'DOCUMENT', 'SAP_NEB.20240715')
                );
            END IF;
        END;
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(create_table_sql)
    cursor.close()  

@app.route('/refresh_clusters', methods=['POST'])
@require_oauth
def refresh_clusters():
    # # Retrieve start_date and end_date from URL arguments
    # start_date = request.args.get('start_date', '1900-01-01')  # Default to '1900-01-01' if not provided
    # end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))  # Default to current date if not provided
    

    # Retrieve start_date and end_date from the form data
    start_date = request.form.get('start_date', '1900-01-01')  # Default to '1900-01-01' if not provided
    end_date = request.form.get('end_date', datetime.now().strftime('%Y-%m-%d'))  # Default to current date if not provided
    
    # Ensure the CLUSTERING table exists
    create_clustering_table_if_not_exists()
    
    # Perform clustering and t-SNE on the ADVISORIES table
    df_clusters, labels = kmeans_and_tsne(
                            connection,  ## Hana ConnectionContext
                            table_name='ADVISORIES4', 
                            result_table_name='CLUSTERING', 
                            n_components=64, 
                            perplexity= 5, ## perplexity for T-SNE algorithm  
                            start_date=start_date,
                            end_date=end_date
                        )
    
    # Insert the values of the "labels" variable into the CLUSTERING_DATA table
    cursor = connection.connection.cursor()
    
    # Delete previous clustering run
    cursor.execute("TRUNCATE TABLE CLUSTERING_DATA")

    for cluster_id, cluster_description in labels.items():
        insert_sql = f"""
            INSERT INTO CLUSTERING_DATA (CLUSTER_ID, CLUSTER_DESCRIPTION)
            VALUES ({cluster_id}, '{cluster_description.replace("'", "''")}')
        """
        cursor.execute(insert_sql)
    cursor.close()

    return jsonify({"message": "Clusters refreshed successfully"}), 200

@app.route('/get_clusters', methods=['GET'])
@require_oauth
def get_clusters():
    # Ensure the CLUSTERING table exists
    create_clustering_table_if_not_exists()
    
    # Retrieve data from the CLUSTERING table
    sql_query = "SELECT * FROM CLUSTERING"
    hana_df = dataframe.DataFrame(connection, sql_query)
    clusters = hana_df.collect()  # Return results as a pandas DataFrame
    
    # Convert DataFrame to list of dictionaries
    formatted_clusters = [
        {
            "x": row["x"],
            "y": row["y"],
            "CLUSTER_ID": row["CLUSTER_ID"],
            "PROJECT_NUMBER": row["PROJECT_NUMBER"]
        }
        for _, row in clusters.iterrows()
    ]
    
    return jsonify(formatted_clusters), 200

@app.route('/get_clusters_description', methods=['GET'])
@require_oauth
def get_clusters_description():
    # Ensure the CLUSTERING table exists
    create_clustering_table_if_not_exists()
    
    # Retrieve data from the CLUSTERING table
    sql_query = "SELECT * FROM CLUSTERING_DATA"
    hana_df = dataframe.DataFrame(connection, sql_query)
    clusters = hana_df.collect()  # Return results as a pandas DataFrame
    
    # Convert DataFrame to list of dictionaries
    formatted_cluster_description = [
        {
            "CLUSTER_ID": row["CLUSTER_ID"],
            "CLUSTER_DESCRIPTION": row["CLUSTER_DESCRIPTION"]
        }
        for _, row in clusters.iterrows()
    ]
    
    return jsonify(formatted_cluster_description), 200

@app.route('/get_projects_by_architect_and_cluster', methods=['GET'])
@require_oauth
def get_projects_by_architect_and_cluster():
    # Retrieve the architect parameter from the URL
    expert = request.args.get('expert')
    
    # Base SQL query
    sql_query = """
        SELECT a."architect", c."CLUSTER_ID", COUNT(a."project_number") AS project_count
        FROM "CLUSTERING" c
        JOIN "ADVISORIES4" a ON c."PROJECT_NUMBER" = a."project_number"
    """
    
    # Add WHERE clause if architect is provided
    if expert:
        sql_query += f"""
        WHERE a."architect" = '{expert.replace("'", "''")}'
        """
    
    # Add GROUP BY clause
    sql_query += """
        GROUP BY a."architect", c."CLUSTER_ID"
    """
    
    hana_df = dataframe.DataFrame(connection, sql_query)
    projects_by_architect_and_cluster = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = projects_by_architect_and_cluster.to_dict(orient='records')
    return jsonify({"projects_by_architect_and_cluster": results}), 200

# Step 2: Function to create the table if it doesn't exist
def create_table_if_not_exists(schema_name, table_name):
    create_table_sql = f"""
        DO BEGIN
            DECLARE table_exists INT;
            SELECT COUNT(*) INTO table_exists
            FROM SYS.TABLES 
            WHERE TABLE_NAME = '{table_name.upper()}' AND SCHEMA_NAME = '{schema_name.upper()}';
            
            IF table_exists = 0 THEN
                CREATE TABLE {schema_name}.{table_name} (
                    TEXT_ID INT GENERATED BY DEFAULT AS IDENTITY,
                    TEXT NVARCHAR(5000),
                    EMBEDDING REAL_VECTOR GENERATED ALWAYS AS VECTOR_EMBEDDING(TEXT, 'DOCUMENT', 'SAP_NEB.20240715')
                );
            END IF;
        END;
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(create_table_sql)
    cursor.close()  
    
# Step 3: Function to insert text and its embedding vector into the "TCM_SAMPLE" table
@app.route('/insert_text_and_vector', methods=['POST'])
@require_oauth
def insert_text_and_vector():

    data = request.get_json()
    schema_name = data.get('schema_name', 'DBUSER')  # Default schema
    table_name = data.get('table_name', 'TCM_SAMPLE')  # Default table
    text = data.get('text')
    # text_type = data.get('text_type', 'DOCUMENT')
    # model_version = data.get('model_version', 'SAP_NEB.20240715')

    # Create the table if it doesn't exist
    create_table_if_not_exists(schema_name, table_name)
    
    # Generate the embedding vector using VECTOR_EMBEDDING
    sql_insert = f"""
        INSERT INTO {schema_name}.{table_name} (TEXT) SELECT '{text}' FROM DUMMY
    """
    
    # Use cursor to execute the query
    cursor = connection.connection.cursor()
    cursor.execute(sql_insert)
    cursor.close()  
    
    return jsonify({"message": f"Text inserted successfully into {schema_name}.{table_name}"}), 200

# Function to compare a new text's vector to existing stored vectors using COSINE_SIMILARITY
@app.route('/compare_text_to_existing', methods=['POST'])
@require_oauth
def compare_text_to_existing():
    data = request.get_json()
    schema_name = data.get('schema_name', 'DBUSER')  # Default schema
    query_text = data.get('query_text')
    text_type = data.get('text_type', 'QUERY')
    model_version = data.get('model_version', 'SAP_NEB.20240715')
    
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400
    
    # Generate the new text's embedding and compare using COSINE_SIMILARITY
    sql_query = f"""
        SELECT "solution" AS text,
               "project_number", 
               COSINE_SIMILARITY(
                   "solution_embedding", 
                   VECTOR_EMBEDDING('{query_text}', '{text_type}', '{model_version}')
               ) AS similarity
        FROM {schema_name}.ADVISORIES4
        UNION ALL
        SELECT "comment" AS text, 
               "project_number", 
               COSINE_SIMILARITY(
                   "comment_embedding", 
                   VECTOR_EMBEDDING('{query_text}', '{text_type}', '{model_version}')
               ) AS similarity
        FROM {schema_name}.COMMENTS4
        ORDER BY similarity DESC
        LIMIT 5
    """
    hana_df = dataframe.DataFrame(connection, sql_query)
    similarities = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = similarities.to_dict(orient='records')
    return jsonify({"similarities": results}), 200

@app.route('/get_project_details', methods=['GET'])
@require_oauth
def get_project_details():
    schema_name = request.args.get('schema_name', 'DBUSER')
    project_number = request.args.get('project_number')
    
    if not project_number:
        return jsonify({"error": "Project number is required"}), 400
    
    # SQL query to join ADVISORIES and COMMENTS tables on project_number
    sql_query = f"""
        SELECT a."architect", a."index" AS advisories_index, a."pbc_number", a."project_date", 
               a."project_number", a."solution", a."topic",
               c."comment", c."comment_date", c."index" AS comments_index
        FROM {schema_name}.advisories4 a
        LEFT JOIN {schema_name}.COMMENTS4 c
        ON a."project_number" = c."project_number"
        WHERE a."project_number" = {project_number}
    """
    hana_df = dataframe.DataFrame(connection, sql_query)
    project_details = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = project_details.to_dict(orient='records')
    return jsonify({"project_details": results}), 200

@app.route('/get_all_projects', methods=['GET'])
@require_oauth
def get_all_projects():
    schema_name = request.args.get('schema_name', 'DBUSER')  # Default schema
    
    # SQL query to retrieve all data from ADVISORIES and COMMENTS tables
    sql_query = f"""
        SELECT * FROM (
            SELECT a."architect", a."index" AS advisories_index, a."pbc_number", a."project_date", 
                   a."project_number", a."solution", a."topic",
                   c."comment", c."comment_date", c."index" AS comments_index,
                   ROW_NUMBER() OVER (PARTITION BY a."project_number" ORDER BY a."index") AS row_num
            FROM {schema_name}.advisories4 a
            LEFT JOIN {schema_name}.COMMENTS4 c
            ON a."project_number" = c."project_number"
        ) subquery
        WHERE row_num = 1
    """
    hana_df = dataframe.DataFrame(connection, sql_query)
    all_projects = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = all_projects.to_dict(orient='records')
    return jsonify({"all_projects": results}), 200

def translate_nl_to_new_helper(nl_query):
    cursor = connection.connection.cursor()
    cursor.execute("""
        SELECT ONTOLOGY_QUERY, PROPERTY_QUERY, CLASSES_QUERY, INSTRUCTIONS, PREFIXES, GRAPH, GRAPH_INFERRED, QUERY_EXAMPLE, TEMPLATE, TEMPLATE_SIMILARITY, QUERY_TEMPLATE, QUERY_TEMPLATE_NO_TOPIC 
        FROM ONTOLOGY_CONFIG
    """)
    config = cursor.fetchone()
    ontology_query = config[0]
    property_query = config[1]
    classes_query = config[2]
    instructions = config[3]
    prefixes = config[4]
    graph = config[5]
    graph_inferred = config[6]
    query_example = config[7]
    template = config[8]
    template_similarity = config[9]
    query_template = config[10]
    query_template_no_topic = config[11]

    # GET ONTOLOGY
    cursor = connection.connection.cursor()
    result = cursor.callproc('SPARQL_EXECUTE', (ontology_query, 'application/sparql-results+csv', '?', '?'))
    ontology = result[2]

    # GET PROPERTIES
    cursor = connection.connection.cursor()
    result = cursor.callproc('SPARQL_EXECUTE', (property_query, 'application/sparql-results+json', '?', '?'))
    properties = result[2]

    # GET CLASSES
    cursor = connection.connection.cursor()
    result = cursor.callproc('SPARQL_EXECUTE', (classes_query, 'application/sparql-results+json', '?', '?'))
    classes = result[0]

    # Topic extraction
    prompt_template_topic = PromptTemplate(
        input_variables=["question"],
        template=template_similarity
    )
    chain_topic = prompt_template_topic | llm | StrOutputParser()
    response_topic = chain_topic.invoke({'question': nl_query})
    response_topic = response_topic.strip('```python\n').strip('\n```')
    response_topic = json.loads(response_topic)
    topic = response_topic["topic"]
    query = response_topic["query"]

    # SPARQL query generation
    prompt_template_sparql = PromptTemplate(
        input_variables=["nl_query", "classes", "properties", "ontology", "graph", "graph_inferred", "prefixes", "query_example", "instructions"],
        template=template
    )
    chain_sparql = prompt_template_sparql | llm
    response_sparql = chain_sparql.invoke({
        "nl_query": query,
        "classes": classes,
        "properties": properties,
        "ontology": ontology,
        "graph": graph,
        "graph_inferred": graph_inferred,
        "prefixes": prefixes,
        "query_example": query_example,
        "instructions": instructions
    })
    sparql_query = response_sparql.content.strip()

    if topic != "None":
        final_query = format_sql(query_template.format(generated_sparql_query=sparql_query, topic=topic))
    else:
        final_query = query_template_no_topic.format(generated_sparql_query=sparql_query)
    return final_query

def execute_query_raw_helper(query, query_type='sparql', response_format='json'):
    cursor = connection.connection.cursor()
    if query_type == 'sparql':
        if response_format == 'csv':
            result = cursor.callproc('SPARQL_EXECUTE', (query, 'application/sparql-results+csv', '?', '?'))
            return {'csv': result[2]}
        else:
            result = cursor.callproc('SPARQL_EXECUTE', (query, 'application/sparql-results+json', '?', '?'))
            return json.loads(result[2])
    elif query_type == 'sql':
        cursor.execute(query)
        if response_format == 'csv':
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]
            csv_data = ','.join(headers) + '\n'
            csv_data += '\n'.join([','.join(map(str, row)) for row in rows])
            return {'csv': csv_data}
        else:
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]
            return [dict(zip(headers, row)) for row in rows]
    else:
        raise ValueError('Invalid query_type. Use "sparql" or "sql".')

@app.route('/execute_query_raw', methods=['POST'])
@require_oauth
def execute_query_raw():
    try:
        query = request.data.decode('utf-8')
        query_type = request.args.get('query_type', 'sparql')
        response_format = request.args.get('format', 'json')
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        result = execute_query_raw_helper(query, query_type, response_format)
        if isinstance(result, dict) and 'csv' in result:
            return Response(result['csv'], mimetype='text/csv')
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/execute_sparql_query', methods=['GET'])
@require_oauth
def execute_sparql_query():
    try:
        # Get the raw SQL query and format from the URL arguments
        query = request.args.get('query')
        response_format = request.args.get('format', 'json')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        cursor = connection.connection.cursor()
        if response_format == 'csv':
            result = cursor.callproc('SPARQL_EXECUTE', (query, 'application/sparql-results+csv', '?', '?'))
            result_csv = result[2]
            return Response(result_csv, mimetype='text/csv')
        else:
            result = cursor.callproc('SPARQL_EXECUTE', (query, 'application/sparql-results+json', '?', '?'))
            result_json = result[2]
            return jsonify(json.loads(result_json)), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/translate_nl_to_sparql', methods=['POST'])
@require_oauth
def translate_nl_to_sparql():
    try:    
        # Get the natural language query and ontology from the request body
        data = request.get_json()
        nl_query = data.get('nl_query')
        
        if not nl_query:
            return jsonify({'error': 'Natural language query required'}), 400

        # Retrieve the configuration from the database
        cursor = connection.connection.cursor()
        cursor.execute("SELECT ONTOLOGY_QUERY, PROPERTY_QUERY, CLASSES_QUERY, INSTRUCTIONS, PREFIXES, GRAPH, GRAPH_INFERRED, QUERY_EXAMPLE, TEMPLATE FROM ONTOLOGY_CONFIG")
        config = cursor.fetchone()

        ontology_query = config[0]
        property_query = config[1]
        classes_query = config[2]
        instructions = config[3]
        prefixes = config[4]
        graph = config[5]
        graph_inferred = config[6]
        query_example = config[7]
        template_config = config[8]

        # GET ONTOLOGY - Directly call the logic of execute_sparql_query
        cursor = connection.connection.cursor()
        result = cursor.callproc('SPARQL_EXECUTE', (ontology_query, 'application/sparql-results+csv', '?', '?'))
        ontology = result[2]

        # GET PROPERTIES - Directly call the logic of execute_sparql_query
        cursor = connection.connection.cursor()
        result = cursor.callproc('SPARQL_EXECUTE', (property_query, 'application/sparql-results+json', '?', '?'))
        properties = result[2]
        
        # GET CLASSES - Directly call the logic of execute_sparql_query
        cursor = connection.connection.cursor()
        result = cursor.callproc('SPARQL_EXECUTE', (classes_query, 'application/sparql-results+json', '?', '?'))
        classes = result[0]
        
        # Define the prompt template
        prompt_template = PromptTemplate(
            input_variables=["nl_query", "classes", "properties", "ontology", "graph", "graph_inferred", "prefixes", "query_example", "instructions"],
            template=template_config
        )

        # Create the LLM chain
        chain = prompt_template | llm
        
        # Run the chain with the provided inputs
        response = chain.invoke({"nl_query": nl_query, 
                                 "classes":classes, 
                                 "properties": properties, 
                                 "ontology": ontology, 
                                 "graph":graph, 
                                 "graph_inferred":graph_inferred, 
                                 "prefixes":prefixes, 
                                 "query_example":query_example, 
                                 "instructions":instructions})
        
        print("response.content: ", response.content)
        sparql_query = response.content.strip()
        print("sparql_query: ", sparql_query)

        return jsonify({'sparql_query': sparql_query}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/translate_nl_to_new', methods=['POST'])
@require_oauth
def translate_nl_to_new():
    try:
        data = request.get_json()
        nl_query = data.get('nl_query')
        if not nl_query:
            return jsonify({'error': 'Natural language query required'}), 400
        final_query = translate_nl_to_new_helper(nl_query)
        return jsonify({'final_query': final_query}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/translate_and_execute', methods=['POST'])
@require_oauth
def translate_and_execute():
    try:
        data = request.get_json()
        nl_query = data.get('nl_query')
        if not nl_query:
            return jsonify({"error": "Natural language query required"}), 400

        # Step 1: Translate NL to SQL/SPARQL
        final_query = translate_nl_to_new_helper(nl_query)

        # Step 2: Execute the generated query and get a list
        result = execute_query_raw_helper(final_query, query_type='sql', response_format='json')

        # Ensure the result is a list for the UI
        if not isinstance(result, list):
            result = [result]

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/config', methods=['GET', 'POST'])
@require_oauth
def config():
    cursor = connection.connection.cursor()
    
    if request.method == 'POST':
        # Update the configuration values
        data = request.get_json()
        ontology_query = data.get('ontology_query')
        property_query = data.get('property_query')
        classes_query = data.get('classes_query')
        instructions = data.get('instructions')
        prefixes = data.get('prefixes')
        graph = data.get('graph')
        graph_inferred = data.get('graph_inferred')
        query_example = data.get('query_example')
        template = data.get('template')
        query_template = data.get('query_template')
        query_template_no_topic = data.get('query_template_no_topic')
        template_similarity = data.get('template_similarity')

        update_query = """
        UPDATE ontology_config SET 
            ontology_query = ?, 
            property_query = ?, 
            classes_query = ?, 
            instructions = ?, 
            prefixes = ?, 
            graph = ?, 
            graph_inferred = ?, 
            query_example = ?,
            template = ?,
            query_template = ?,
            query_template_no_topic = ?,
            template_similarity = ?
        """
        cursor.execute(update_query, (ontology_query, property_query, classes_query, instructions, prefixes, graph, graph_inferred, query_example, template, query_template, query_template_no_topic, template_similarity))
        connection.connection.commit()
        return jsonify({'message': 'Configuration updated successfully'}), 200

    # Retrieve the current configuration values
    cursor.execute("SELECT ONTOLOGY_QUERY, PROPERTY_QUERY, CLASSES_QUERY, INSTRUCTIONS, PREFIXES, GRAPH, GRAPH_INFERRED, QUERY_EXAMPLE, TEMPLATE, QUERY_TEMPLATE, QUERY_TEMPLATE_NO_TOPIC, TEMPLATE_SIMILARITY FROM ONTOLOGY_CONFIG")
    config = cursor.fetchone()
    return jsonify({
        'ontology_query': config[0],
        'property_query': config[1],
        'classes_query': config[2],
        'instructions': config[3],
        'prefixes': config[4],
        'graph': config[5],
        'graph_inferred': config[6],
        'query_example': config[7],
        'template': config[8],
        'query_template': config[9],
        'query_template_no_topic': config[10],
        'template_similarity': config[11]
    }), 200
    
@app.route('/get_advisories_by_expert_and_category', methods=['GET'])
@require_oauth
def get_advisories_by_expert_and_category():
    expert = request.args.get('expert')
    
    if not expert:
        return jsonify({"error": "Expert is required"}), 400
    
    # SQL query to retrieve the number of advisories by expert and category
    sql_query = f"""
        SELECT c."category_label" AS category, COUNT(a."project_number") AS projects
        FROM "PROJECT_BY_CATEGORY" pbc
        JOIN "CATEGORIES" c ON pbc."CATEGORY_ID" = c."index"
        JOIN "ADVISORIES4" a ON pbc."PROJECT_ID" = a."project_number"
        WHERE a."architect" = '{expert.replace("'", "''")}'
        GROUP BY c."category_label"
    """
    hana_df = dataframe.DataFrame(connection, sql_query)
    advisories_by_category = hana_df.collect()  # Return results as a pandas DataFrame

    # Convert results to a list of dictionaries for JSON response
    results = advisories_by_category.to_dict(orient='records')
    return jsonify({"advisories_by_category": results}), 200

@app.route('/openapi.json', methods=['GET'])
@require_oauth
def openapi_spec():
    with open(os.path.join(os.path.dirname(__file__), 'specOpenAPI.json')) as f:
        spec = json.load(f)
    return jsonify(spec)

@app.route('/', methods=['GET'])
def root():
    return 'Embeddings API: Health Check Successfull.', 200

def create_app():
    return app

# Start the Flask app
if __name__ == '__main__':
    app.run('0.0.0.0', 8080)