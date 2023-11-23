SELECT iddia, COUNT(*) AS n FROM public.sm_tablets GROUP BY iddia;
SELECT iddia, COUNT(*) AS n FROM public.sm_tablets_obs GROUP BY iddia;

SELECT "fecha_corte", COUNT(*) AS n FROM public.obs_recepcion_bi GROUP BY "fecha_corte";
SELECT "fecha_corte", COUNT(*) AS n FROM public.obs_asigna_bi GROUP BY "fecha_corte";

SELECT "idcatasignacion", COUNT(*) AS n FROM public.obs_asigna_bi WHERE "fecha_corte"='2023-11-08' GROUP BY "idcatasignacion";
SELECT "tipo_categoria", COUNT(*) AS n FROM public.obs_asigna_bi WHERE "fecha_corte"='2023-11-08' GROUP BY "tipo_categoria";


SELECT * FROM public."SIAGIE2023" LIMIT 100;
SELECT * FROM public."export_traza_use" LIMIT 100;
SELECT * FROM public."obs_recepcion";
SELECT * FROM public."obs_recepcion_bi";
SELECT * FROM public."obs_asigna";
SELECT * FROM public."obs_asigna_bi";

SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'obs_asigna_bi';


SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'sm_tablets';

SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'sm_tablets_obs';

SELECT MAX(DISTINCT iddia) FROM public."sm_tablets";

SELECT * FROM public."sm_tablets_obs" LIMIT 100;
-----------
DELETE FROM public.sm_tablets_obs WHERE iddia = 20231025;
DELETE FROM public.sm_tablets_obs WHERE iddia = 20231025;

INSERT INTO public.sm_tablets_obs
SELECT *
FROM public.sm_tablets
WHERE iddia = (SELECT MAX(DISTINCT iddia) FROM public.sm_tablets) --20231108
AND (LENGTH("OBSERVACION_RECEPCION") >0 OR
LENGTH("OBSERVACION_EQUIPO") >0 OR
LENGTH("OBSERVACIONES_A") >0 OR
LENGTH("OBSERVACIONES_PERDIDA") >0);



SELECT DISTINCT A."CODIGO_MODULAR",
				A."NRO_PECOSA",
			 	A."OBSERVACION_RECEPCION",
			 	A."FECHA_CREACION_R",
			 	A."FECHA_MODIFICACION_R",
			 	A."FASE",
			 	B."idcatrecepcion"
		FROM public."sm_tablets_obs" A
		LEFT JOIN (SELECT * FROM public."obs_recepcion_bi" 
				   WHERE "fecha_corte"= (SELECT MIN (DISTINCT fecha_corte) FROM public.obs_recepcion_bi)
				   ) B
		ON (A."CODIGO_MODULAR" = B."idinstitucioneducativa" AND
			A."OBSERVACION_RECEPCION" = B."OBSERVACION_RECEPCION")
		WHERE LENGTH(A."OBSERVACION_RECEPCION")>0 
		AND A."iddia"=(SELECT MAX (DISTINCT iddia) FROM public.sm_tablets_obs)
		AND B."idcatrecepcion" IS NULL;


