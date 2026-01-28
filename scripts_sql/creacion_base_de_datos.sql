-- Crear las bases de datos
use master;
go

-- Registros estaciones meteorológicas: 
if not exists (
    select name
    from sys.databases
    where name = N'BD_REM'
)
begin
    print 'Creando BD BD_REM'
    create database BD_REM;
end
else
begin
    print 'La BD BD_REM ya existe'
end
go
