use WSPresionAtm;
go

-- CREACION DE LA TABLA PRESIÓN --
-- Se eliminan las tablas previamente creadas

if object_id(N'dbo.presion', 'U') is not null drop table dbo.presion;

-- Se crean las tablas
create table dbo.presion (
	tx_id				int identity(1,1) not null primary key,
	codigo_estacion		varchar(100),
	codigo_sensor		varchar(100),
	fecha_observacion	datetime,
	valor_observado		float,
	nombre_estacion		varchar(100),
	departamento		varchar(100),
	municipio			varchar(100),
	zona_hidrografica	varchar(100),
	latitud				float,
	longitud			float,
	descripcion_sensor	varchar(100),
	unidad_medida		varchar(100),
);
go
