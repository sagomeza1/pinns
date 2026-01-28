use WSVelocidadVien;
go

-- CREACION DE LA TABLA PRESI�N --
-- Se eliminan las tablas previamente creadas

if object_id(N'dbo.vel_viento', 'U') is not null drop table dbo.vel_viento;

-- Se crean las tablas
create table dbo.vel_viento (
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
