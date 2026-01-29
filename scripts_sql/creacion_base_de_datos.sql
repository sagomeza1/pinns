-- Crear las bases de datos
use master;
go

-- Registros estaciones meteorológicas: 
if not exists (
    select name
    from sys.databases
    where name = N'EM_CUNBOY'
)
begin
    print 'Creando BD EM_CUNBOY'
    create database EM_CUNBOY;
end
else
begin
    print 'La BD EM_CUNBOY ya existe'
end
go
