use EM_CUNBOY;
go


with estacion_full as (
	select 
		pvd.codigo_estacion,
		pvd.fecha_observacion,
		pvd.sensor_pre,
		pvd.sensor_vel,
		pvd.sensor_dir,
		t.codigo_sensor as sensor_tem,
		pvd.presion,
		pvd.velocidad,
		pvd.direccion,
		t.valor_observado as temperatura
	from (
		select
			pv.codigo_estacion,
			pv.fecha_observacion,
			pv.sensor_pre,
			pv.sensor_vel,
			pv.presion,
			pv.velocidad,
			d.valor_observado as direccion,
			d.codigo_sensor as sensor_dir
		from (
			select
				p.codigo_estacion,
				p.fecha_observacion,
				p.codigo_sensor as sensor_pre,
				v.codigo_sensor as sensor_vel,
				p.valor_observado as presion,
				v.valor_observado as velocidad
				from dbo.presion as p
			inner join dbo.vel_viento as v
				on p.codigo_estacion = v.codigo_estacion
				and p.fecha_observacion = v.fecha_observacion
		) as pv
		inner join dbo.dir_viento as d
			on pv.codigo_estacion = d.codigo_estacion
			and pv.fecha_observacion = d.fecha_observacion
	) as pvd
	inner join dbo.temp_aire as t
		on pvd.codigo_estacion = t.codigo_estacion
		and pvd.fecha_observacion = t.fecha_observacion
)

insert into dbo.estaciones_full (
			codigo_estacion,
			fecha_observacion,
			sensor_pre,
			sensor_vel,
			sensor_dir,
			sensor_tem,
			presion,
			velocidad,
			direccion,
			temperatura
)
select *
from estacion_full;

