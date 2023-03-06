close all
clear

tic
% addpath Tetra2VoxelConversion
addpath VoxelPlotter
result_folder='Resultant_Folder_128_128_whcp_mri2mesh';
example=3;
epoch='1660';


file_vox=strcat('./',result_folder,'/Inference_Result/epoch_',string(epoch),'/example_',string(example)','.mat');
load(file_vox);
voxelRegions=voxelRegions.*5;
voxelRegions=reshape(voxelRegions,[128,128,128]);
voxelRegions=ceil(voxelRegions);
voxelConductivity=zeros(128,128,128);
cond=[0.1,0.2,0.3,0.4,0.5];
for i=1:5
    voxelConductivity(voxelRegions==i)=cond(i);
end
dx=1;
%Plot Conductivity
cond={'WM','GM','CSF','Skull','Skulp'};
count=1;
for i=1:5
    figure(i)
    vox=voxelConductivity;
    colormap jet
    for j=1:5
        if i~= j 
            vox(voxelRegions==j)=0;
        end
    end
    [vol_handle]=VoxelPlotter(vox,dx); 
    colorbar
    material(vol_handle,'dull');
    lighting gouraud
    camlight('left');
    grid on
    title(strcat('Conductivity - ',cond{count}));
    view(3);
    %axis([0 144 0 144 0 144]);
    fig=figure(i);
    save_dir = strcat('./',result_folder,'/Analysis/epoch_',string(epoch));
    if ~exist(save_dir, 'dir')
        mkdir(save_dir)
    end
    saveas(fig,strcat(save_dir,'/example_',string(example),'_conductivity_',cond{count}));
    count=count+1;
end


toc
close all
return
