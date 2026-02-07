-- Crear las bases de datos
use master;
go

-- Registros estaciones meteorológicas: 
if not exists (
    select name
    from sys.databases
    where name = N'EM_CAR'
)
begin
    print 'Creando BD EM_CAR'
    create database EM_CAR;
end
else
begin
    print 'La BD EM_CAR ya existe'
end
go
