from common import DEBUG,Image,plt,data,iaa,np,device,torch,os,det_args,rec_args,cv2
from detector.model import train_processes
from detector.detect import det_train
from detector.preprocess import ImageDataset,NormalizeImage
from detector.test import det_test
from identifier.identify import rec_train
from identifier.model import RecModelBuilder
from identifier.postprocess import final_postProcess
from identifier.preprocesss import IdentifierPreProcess, WMRDataset
from identifier.test import rec_test, rec_test_data_gen
 
'''
检测模型
'''
    
# 数据处理可视化
train_dataset = ImageDataset(data_dir=det_args.train_dir, gt_dir=det_args.train_gt_dir, is_training=True, processes=train_processes)
train_dataloader = data.DataLoader(train_dataset, batch_size=2, num_workers=0, shuffle=True, drop_last=False)
batch = next(iter(train_dataloader))    # 获取一个 batch
 
 
# 画图
plt.figure(figsize=(60,60))
image = NormalizeImage.restore(batch['image'][0])
plt.subplot(141)
plt.title('image', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(image)
 
 
probability_map = (batch['gt'][0].to('cpu').numpy() * 255).astype(np.uint8)
plt.subplot(142)
plt.title('probability_map', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(probability_map, cmap='gray')
 
 
threshold_map = (batch['thresh_map'][0].to('cpu').numpy() * 255).astype(np.uint8)
plt.subplot(143)
plt.title('threshold_map', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(threshold_map, cmap='gray')
 
 
threshold_mask = (batch['thresh_mask'][0].to('cpu').numpy() * 255).astype(np.uint8)
plt.subplot(144)
plt.title('threshold_mask', fontdict={'size': 60})
plt.xticks([])
plt.yticks([])
plt.imshow(threshold_mask, cmap='gray')
 
 
det_train()


''' 
检测模型 测试
'''
if DEBUG:
    det_test()

''' 
检测模型 End
'''
    

'''
识别器
''' 
 
# 运行识别训练数据前处理代码 
IdentifierPreProcess()


dataset = WMRDataset(rec_args.train_dir, max_len=5, resize_shape=(rec_args.height, rec_args.width), train=True)
train_dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, drop_last=False)
batch = next(iter(train_dataloader))
 
 
image, label, label_len = batch
image = ((image[0].permute(1, 2, 0).to('cpu').numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
plt.title('image')
plt.xticks([])
plt.yticks([])
plt.imshow(image)
        
label_digit = label[0].to('cpu').numpy().tolist()
label_str = ''.join([dataset.id2char[t] for t in label_digit if t > 0])
 
 
print('label_digit: ', label_digit)
print('label_str: ', label_str)

'''
模型各阶段数据结构展示
'''
dataset = WMRDataset(rec_args.train_dir, max_len=rec_args.max_len, resize_shape=(rec_args.height, rec_args.width), train=True)
train_dataloader = data.DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True, pin_memory=True, drop_last=False)
batch = next(iter(train_dataloader))
 
 
model = RecModelBuilder(rec_num_classes=rec_args.voc_size, sDim=rec_args.decoder_sdim)
model = model.to(device)
model.train()
 
 
image, rec_targets, rec_lengths = [v.to(device) for v in batch]
encoder_out = model.encoder(image)
decoder_out = model.decoder(encoder_out.transpose(0, 1).contiguous())
 
 
 
# batch 输入
print('batch 输入 [image, label, label_length]：')
print(batch[0].shape)
print(batch[1].shape)
print(batch[2].shape)
print()
 
 
# encoder 输出
print('encoder 输出：')
print(encoder_out.shape)
print()
 
 
# decoder 输出
print('decoder 输出：')
print(decoder_out.shape)

rec_train()


''' 
识别器 测试
'''
if DEBUG:
    rec_test_data_gen()
    rec_test()

'''
识别器 后处理
'''
final_postProcess()

''' 
检测模型 End
'''
