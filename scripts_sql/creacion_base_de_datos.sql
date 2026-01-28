-- Crear las bases de datos
use master;
go

-- Registros estaciones meteorológicas: 
if not exists (
    select name
    from sys.databases
    where name = N'EM_CUN'
)
begin
    print 'Creando BD EM_CUN'
    create database EM_CUN;
end
else
begin
    print 'La BD EM_CUN ya existe'
end
go
