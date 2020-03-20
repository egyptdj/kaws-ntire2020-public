function crop_center(target, source, start_id, end_id)
% NTIRE 2020 - crop function for submission
%
% target - path to output the cropped image
% source - path of the resolved images
% start_id - image id to start cropping
% end_id - image id to end cropping
%
% saves cropped images to the target path

fileids=start_id:end_id;
impath=source;

for id=fileids
  id = num2str(id);
  I = imread(strcat(impath,'/',id,'.png'));
  sz = size(I);
  Ic = imcrop(I, [sz(2)/2-499 sz(1)/2-499 999 999]);
  imwrite(Ic, strcat(target,id,'.png'), 'png');
end
