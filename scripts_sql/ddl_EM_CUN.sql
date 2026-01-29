use EM_CUNBOY;
go

-- Se eliminan las tablas previamente creadas

if object_id(N'dbo.presion', 	'U') is not null drop table dbo.presion;
if object_id(N'dbo.vel_viento', 'U') is not null drop table dbo.vel_viento;
if object_id(N'dbo.dir_viento', 'U') is not null drop table dbo.dir_viento;
if object_id(N'dbo.temp_aire', 'U') is not null drop table dbo.temp_aire;
if object_id(N'dbo.humd_aire', 'U') is not null drop table dbo.humd_aire;
if object_id(N'dbo.estaciones_full', 'U') is not null drop table dbo.estaciones_full;
if object_id(N'dbo.coordenadas_estaciones', 'U') is not null drop table dbo.coordenadas_estaciones;

-- Se crean las tablas

-- CREACION DE LA TABLA PRESION --
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

-- CREACION DE LA DIRECCION DEL VIENTO --

create table dbo.dir_viento (
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

	-- CREACION DE LA VELOCIDAD DEL VIENTO --

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

create table dbo.temp_aire (
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

create table dbo.humd_aire (
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

create table dbo.estaciones_full (
	tx_id				int identity(1,1) not null primary key,
	codigo_estacion		varchar(100),
	fecha_observacion	datetime,
	sensor_pre			varchar(100),
	sensor_vel			varchar(100),
	sensor_dir			varchar(100),
	sensor_tem			varchar(100),
	presion				float,
	velocidad			float,
	direccion			float,
	temperatura			float
);
go

create table dbo.coordenadas_estaciones (
	tx_id				int identity(1,1) not null primary key,
	codigo_estacion		varchar(100),
	longitud			float,
	latitud				float
);
go