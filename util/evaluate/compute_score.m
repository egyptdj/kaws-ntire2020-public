function [psnr, ssim] = compute_score(target, source, start_id, end_id)
% NTIRE 2020 - function for evaluating the psnr and ssim scores
%
% target - path of the label images
% source - path of the resolved images
% start_id - image id to start cropping
% end_id - image id to end cropping
%
% prints and returns psnr, ssim score between the target and the source

fileids=start_id:end_id;
labelpath=target;
reconpath=source;

psnr_sum = 0.0;
ssim_sum = 0.0;
for id=fileids
  id = num2str(id);
  psnr_sum = psnr_sum + NTIRE_PeakSNR_imgs(strcat(labelpath,'/',id,'.png'), strcat(reconpath,'/',id,'.png'), 16);
  ssim_sum = ssim_sum + NTIRE_SSIM_imgs(strcat(labelpath,'/',id,'.png'), strcat(reconpath,'/',id,'.png'), 16);
end

psnr = psnr_sum/(end_id-start_id+1)
ssim = ssim_sum/(end_id-start_id+1)
