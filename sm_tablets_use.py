import pandas as pd
import numpy as np
import pyodbc
from os import chdir, getcwd
from datetime import datetime
#-----------------------------------------------------------------------------*
maindir = "E:/Mi unidad/dgavidia_minedu/BD USE/NLP/TABLETAS"
#-----------------------------------------------------------------------------*
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=med000008404;'
                      'Trusted_Connection=yes;')
#-----------------------------------------------------------------------------*
start = datetime.now()
from datetime import datetime, timedelta
start_mes = start - timedelta(days=30)
mes = start_mes.strftime("%m")
año = start.strftime("%Y")
dia = "01"
print(año+mes)

query_fec = """SELECT TOP 1 iddia 
               FROM [SA_DATA].[dbo].[sa_34SMrecepcion_pecosa_tablet] 
               WHERE iddia >= '""" + año+mes+dia + """'
               ORDER BY iddia DESC"""
               
fec_t          = pd.read_sql_query(query_fec,conn)

fec_t = str(fec_t['iddia'][0])
print(datetime.now() - start)
#-----------------------------------------------------------------------------*
import psycopg2

# Establecer los detalles de conexión
db_params = {
    "dbname": "DGAVIDIA",
    "host": "localhost",
    "user": "postgres",
    "port": 5432,
    "password": "a07119420"
}

# Definir la consulta SQL
query = """INSERT INTO public.sm_tablets_obs
           SELECT *
           FROM public.sm_tablets
           WHERE iddia = (SELECT MAX(DISTINCT iddia) FROM public.sm_tablets) --20231108
           AND (LENGTH("OBSERVACION_RECEPCION") >0 OR
           LENGTH("OBSERVACION_EQUIPO") >0 OR
           LENGTH("OBSERVACIONES_A") >0 OR
           LENGTH("OBSERVACIONES_PERDIDA") >0);"""

# Establecer la conexión y ejecutar la consulta
try:
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    print("Consulta ejecutada exitosamente.")
except psycopg2.Error as e:
    print("Error al conectar a PostgreSQL o al realizar la consulta:", e)
finally:
    conn.close()



                                 
                                          
                                          

                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          