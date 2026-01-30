use EM_BOG;
go

update dbo.presion
set latitud = round(latitud, 3);

update dbo.presion
set longitud = round(longitud, 3);
---------------------------------
update dbo.dir_viento
set latitud = round(latitud, 3);

update dbo.dir_viento
set longitud = round(longitud, 3);
---------------------------------
update dbo.vel_viento
set latitud = round(latitud, 3);

update dbo.vel_viento
set longitud = round(longitud, 3);
---------------------------------
update dbo.temp_aire
set latitud = round(latitud, 3);

update dbo.temp_aire
set longitud = round(longitud, 3);
---------------------------------
