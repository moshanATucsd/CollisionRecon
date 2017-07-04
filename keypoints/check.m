file_xyz = '/home/dinesh/CarCrash/data/Fifth/Corpus/3dGL.xyz';
t = importdata(file_xyz,' ');
points_3d = t(:,1:3);
scatter3(t(:,1),t(:,2),t(:,3));
axis equal;
hold on;
Folder = '/home/dinesh/CarCrash/data/Fifth/';
K = zeros(22,10005,3,3);
for l=1:21
    try
        sd = importdata([Folder  '/vHIntrinsic_' num2str(l-1) '.txt']);
    catch
        continue
    end
    
    for j = 1:size(sd,1)
        K(l,sd(j,1)+1,:,:) = [sd(j,6),0,sd(j,9);0,sd(j,7),sd(j,10);0,0,1];
    end
end
rt = zeros(22,10000,3,4);
for l=1:21
    try
        sd = importdata([Folder  '/vHCamPose_RSCayley_' num2str(l-1) '.txt']);
    catch
        continue
    end
    
    for j = 1:size(sd,1)
        R = rotationVectorToMatrix([sd(j,2),sd(j,3),sd(j,4)]);
        T = [sd(j,5),sd(j,6),sd(j,7)];
        RT = [R,T'];
        rt(l,sd(j,1)+1,:,:) = RT;%[sd(j,6),0,sd(j,9);0,sd(j,7),sd(j,10);0,0,1];
    end
end

sync = importdata([Folder  '/InitSync.txt']);

time = 0;
scatter3(-3.7957,-0.0098,0.1951);
for camera =1:21
    cam_num = time-sync(camera,2);
    [Folder '/' num2str(camera-1)  '/'  num2str(cam_num,'%05d')  '.png']

    try
    save = imread([Folder '/' num2str(camera-1)  '/'  num2str(cam_num,'%05d')  '.png']);
    RT_s = reshape(rt(camera,cam_num,:,:),[3 4]);
    k_s = reshape(K(camera,cam_num,:,:),[3 3]);
    R = RT_s(1:3,1:3);
    T = -RT_s(1:3,1:3) * RT_s(1:3,4);
    cam = plotCamera('Location',T','Orientation',R','Opacity',0,'size',0.1);
    RT_s = [R',RT_s(1:3,4)]
    point_2d = k_s * RT_s*[points_3d(1,1:3),1]';
    point_2d = point_2d/point_2d(3)
    imwrite(save,[num2str(camera,'%05d') '.png'])
    
    figure
    imshow(save);
    hold on;
    scatter(point_2d(1),point_2d(2))
    ginput(1);
    catch
        continue
    end
end


