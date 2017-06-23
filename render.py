import bpy
from mathutils import Vector
import recon_json as recon
import json
import types_re
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance as distance 

def ScaleWorld():
	# scale the world
	scene = bpy.context.scene

	for ob in scene.objects:
		if ob.type == "MESH":
			ob.select = True
	bpy.ops.object.join()	
	for ob in scene.objects:
		if ob.type == "MESH":
			ob.select = False	
	
	maxValue = 0
	for ob in scene.objects:
		centre = sum((Vector(b) for b in ob.bound_box), Vector())
		centre /= 8
		check = ob.dimensions + centre
		if check[0] > maxValue:
			maxValue = check[0]
		if check[1] > maxValue:
			maxValue = check[1]
		if check[2] > maxValue:
			maxValue = check[2]
		
		
	scale = 10/maxValue
	for ob in scene.objects:
		ob.scale = (scale,scale,scale)

def seed_position(CamerPoint,Origin,aspect_ratio):
    
    cam = bpy.data.objects['Camera']
    cam.data.sensor_width=30*aspect_ratio
    cam.data.sensor_height=30
    cam.location.x = CamerPoint[0]
    cam.location.y = CamerPoint[1]
    cam.location.z = CamerPoint[2]
    obj_lamp = bpy.data.objects["Lamp"]
    obj_lamp.location = cam.location
    loc_camera = cam.location
    vec = Vector((Origin[0], Origin[1], Origin[2]))
    direction = vec - loc_camera
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()


# select objects by type
for o in bpy.data.objects:
    if o.type == 'MESH':
        o.select = True
    else:
        o.select = False

# call the operator once
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath="/home/dinesh/CarCrash/data/CarCrash/Cleaned/models/dodge/dodge_scaled_meshed_22k.obj")
scene = bpy.context.scene
filename = '/home/dinesh/CarCrash/data/CarCrash/Cleaned/0/reconstruction.json'
with open(filename) as fin:
	obj = json.loads(fin.read())
#print(obj[0])
reconstructions = recon.reconstructions_from_json(obj)

for j, reconstruction in enumerate(reconstructions):
    points = reconstruction.points
    shots = reconstruction.shots
    cameras = reconstruction.cameras
    num_point = len(points)
    num_shot = len(shots)
    
car_points = []
for point_id, point in points.items():
    coord = point.coordinates
    color = map(int, point.color)
    car_points.append(coord) 
car_points = np.array(car_points)
mean = np.median(car_points,axis=0)
car_points_cleaned = []

for point in car_points:
    if distance.euclidean(point,mean) <20000:
        car_points_cleaned.append(point)
mean_cleaned = np.mean(car_points_cleaned,axis=0)
car_points_cleaned = np.array(car_points_cleaned)
dist_indices = (dist<= np.average(dist))
fig = pylab.figure()
ax = Axes3D(fig)
ax.scatter(car_points_cleaned[:,0], car_points_cleaned[:,1], car_points_cleaned[:,2])


for shot_id in shots:
    shot = shots[shot_id]
    camera = shot.camera
    scale = max(camera.width, camera.height)
    focal = camera.focal * scale
    k1 = camera.k1
    k2 = camera.k2
    cam_width = camera.width
    cam_height = camera.height
    aspect_ratio = cam_width/cam_height
    R = shot.pose.get_rotation_matrix()
    t = np.array(shot.pose.translation)


print(car_points)
for i in range(num_shot):
	for ob in scene.objects:
		if ob.type == 'MESH':
			ob.location = (-10 + i/10,0,15)
			ob.rotation_euler = (0.5,2,2)
	seed_position((0,0,0),(0,0,2),aspect_ratio)
	
	bpy.context.scene.render.resolution_x = cam_width #perhaps set resolution in code
	bpy.context.scene.render.resolution_y = cam_height
	bpy.data.scenes['Scene'].render.filepath = '/home/dinesh/CarCrash/data/CarCrash/Cleaned/0/virtual/model' + str(i)+'.png'
	#bpy.ops.render.render( write_still=True )
