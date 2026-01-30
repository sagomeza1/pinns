-- Crear las bases de datos
use master;
go

-- Registros estaciones meteorológicas: 
if not exists (
    select name
    from sys.databases
    where name = N'EM_BOG'
)
begin
    print 'Creando BD EM_BOG'
    create database EM_BOG;
end
else
begin
    print 'La BD EM_BOG ya existe'
end
go
