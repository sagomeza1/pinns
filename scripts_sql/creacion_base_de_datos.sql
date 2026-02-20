-- Crear las bases de datos
use master;
go

-- Registros estaciones meteorológicas: 
if not exists (
    select name
    from sys.databases
    where name = N'EM_CAR3'
)
begin
    print 'Creando BD EM_CAR3'
    create database EM_CAR3;
end
else
begin
    print 'La BD EM_CAR3 ya existe'
end
go
