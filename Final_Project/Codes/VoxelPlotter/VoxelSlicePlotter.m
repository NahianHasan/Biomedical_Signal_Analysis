function [vol_handle]=VoxelSlicePlotter(VoxelMat,Vox_Size)
%detect the external voxels and faces
vol_handle=0;
if nargin==1
Vox_Size=1;
end
    FV=FindVoxelSlice(VoxelMat,Vox_Size);
%plot only external faces of external voxels
cla;
if size(FV.vertices,1)==0
    cla;
else
vol_handle=patch(FV,'EdgeColor','none');
%use patchslim here for better results
end
end

