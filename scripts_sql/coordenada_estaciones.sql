use EM_BOG;
go

with lon_lat as(
    select codigo_estacion , longitud, latitud
    from dbo.presion
    union
    select codigo_estacion , longitud, latitud
    from dbo.dir_viento
    union
    select codigo_estacion , longitud, latitud
    from dbo.vel_viento
    union
    select codigo_estacion , longitud, latitud
    from dbo.temp_aire
) 

insert into dbo.coordenadas_estaciones (codigo_estacion, longitud, latitud)
    select codigo_estacion , min(longitud) as lon , min(latitud) as lat
    from lon_lat group by codigo_estacion