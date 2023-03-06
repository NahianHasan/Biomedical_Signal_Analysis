function only_voxel_data_generation(output_mesh_folder,output_mesh_file,output_voxel_folder,cond)

tstart=tic;

addpath VoxelPlotter

mtest_mean=mesh_load_gmsh4(strcat(output_mesh_folder,output_mesh_file));

tetFieldsTotal = mesh_extract_regions(mtest_mean,'elemtype','tet','region_idx', [1 2 3 4 5 6 7 8 501 502]);
nodePointsTotal = tetFieldsTotal.nodes ;
JfieldsTetra = tetFieldsTotal.element_data{2,1}.tetdata;
tetraRegions = double(tetFieldsTotal.tetrahedron_regions);

%%%%%%%%%%% GEnerate the Grid Points %%%%%%%%%%%%%%%%%%%%%
xLength=max(nodePointsTotal(:,1))-min(nodePointsTotal(:,1));
yLength=max(nodePointsTotal(:,2))-min(nodePointsTotal(:,2));
zLength=max(nodePointsTotal(:,3))-min(nodePointsTotal(:,3));

maxEdge=max([xLength yLength zLength]) ;
dx=ceil(maxEdge)/144;
dcell=[dx;dx;dx];

[XX,YY,ZZ]=ndgrid(min(nodePointsTotal(:,1)):dcell(1):max(nodePointsTotal(:,1)),...
                  min(nodePointsTotal(:,2)):dcell(2):max(nodePointsTotal(:,2)),...
                  min(nodePointsTotal(:,3)):dcell(3):max(nodePointsTotal(:,3)));% cell node locations

robs(:,3)=ZZ(:)'+dcell(3)/2;
robs(:,2)=YY(:)'+dcell(2)/2;
robs(:,1)=XX(:)'+dcell(1)/2;
 

%%%%%%%% get J field at each voxel Point %%%%%%%%%%
TR=triangulation(tetFieldsTotal.tetrahedra,nodePointsTotal);
teid=pointLocation(TR,robs(:,1),robs(:,2),robs(:,3));
JFields=zeros(size(robs));
RegionsVoxel=zeros(size(robs(:,1)));
JFields((isnan(teid)==0),:)=JfieldsTetra(teid(isnan(teid)==0),:);
RegionsVoxel((isnan(teid)==0),:)=tetraRegions(teid(isnan(teid)==0));

JFields = reshape(JFields,[size(XX) 3]);
RegionsVoxel = reshape(RegionsVoxel, [size(XX) 1]);

bufferX = ceil((144-size(RegionsVoxel,1))/2);
bufferY = ceil((144-size(RegionsVoxel,2))/2);
bufferZ = ceil((144-size(RegionsVoxel,3))/2);

headVoxelJfield=zeros(144,144,144,3);
headVoxelJfield(bufferX+1:bufferX+size(RegionsVoxel,1),...
    bufferY+1:bufferY+size(RegionsVoxel,2),bufferZ+1:bufferZ+size(RegionsVoxel,3),1:3)=...
    JFields(:,:,:,1:3);

voxelRegions = zeros(144,144,144);
voxelRegions(bufferX+1:bufferX+size(RegionsVoxel,1),...
    bufferY+1:bufferY+size(RegionsVoxel,2),bufferZ+1:bufferZ+size(RegionsVoxel,3))=...
    RegionsVoxel(:,:,:);

voxelConductivity=zeros(144,144,144);
for i=1:6
    voxelConductivity(voxelRegions==i)=cond(i);
end
voxelConductivity(voxelRegions==501)=2;
voxelConductivity(voxelRegions==502)=3;

save(strcat(output_voxel_folder,'field_cond.mat'),'headVoxelJfield','voxelConductivity','voxelRegions');
fprintf('Generated voxels in %f seconds\n',toc(tstart))
end
