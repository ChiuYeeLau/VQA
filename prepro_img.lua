require 'nn'
require 'optim'
require 'torch'
require 'nn'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
require 'paths'
require 'cudnn'
cjson=require('cjson')
require 'xlua'

local t = require 'transforms'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_json','data_prepro.json','path to the json file containing vocab and answers')
cmd:option('-image_root','','path to the image root')
--cmd:option('-cnn_proto', '', 'path to the cnn prototxt')
cmd:option('-cnn_model', '', 'path to the cnn model')
cmd:option('-batch_size', 10, 'batch_size')

cmd:option('-out_name', 'data_img_resnet.h5', 'output name')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

local batch_size = opt.batch_size

cutorch.setDevice(opt.gpuid)

local model = torch.load(opt.cnn_model):cuda()

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local image_root = opt.image_root

local list_of_filenames_train = {}
for i,imname in pairs(json_file['unique_img_train']) do
    table.insert(list_of_filenames_train, image_root .. imname)
end

local list_of_filenames_test = {}
for i,imname in pairs(json_file['unique_img_test']) do
    table.insert(list_of_filenames_test, image_root .. imname)
end

local number_of_files_train = #list_of_filenames_train
local number_of_files_test = #list_of_filenames_test

assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
model:remove(#model.modules)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

local features_test, features_train

for i = 1, number_of_files_train,batch_size do

    local img_batch = torch.FloatTensor(batch_size, 3, 224, 224) -- batch numbers are the 3 channels and size of transform

    -- preprocess the images for the batch
    local image_count = 0
    for j=1,batch_size do
        img_name = list_of_filenames_train[i+j-1]

        if img_name  ~= nil then
            image_count = image_count + 1
            local img = image.load(img_name, 3, 'float')
            img = transform(img)
            img_batch[{j, {}, {}, {} }] = img
        end
    end


    -- if this is last batch it may not be the same size, so check that
    if image_count ~= batch_size then
        img_batch = img_batch[{{1,image_count}, {}, {}, {} } ]
    end

   -- Get the output of the layer before the (removed) fully connected layer
    local output = model:forward(img_batch:cuda()):squeeze(1)


    -- this is necesary because the model outputs different dimension based on size of input
    if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end

    if not features_train then
       features_train = torch.FloatTensor(number_of_files_train, output:size(2)):zero()
    end
       features_train[{ {i, i-1+image_count}, {}  } ]:copy(output)

    if (i - math.floor(i/500)*500 == 1) then
        print(i .. "train")
    end

end

for i = 1, number_of_files_test,batch_size do

    local img_batch = torch.FloatTensor(batch_size, 3, 224, 224) -- batch numbers are the 3 channels and size of transform

    -- preprocess the images for the batch
    local image_count = 0
    for j=1,batch_size do
        img_name = list_of_filenames_test[i+j-1]

        if img_name  ~= nil then
            image_count = image_count + 1
            local img = image.load(img_name, 3, 'float')
            img = transform(img)
            img_batch[{j, {}, {}, {} }] = img
        end
    end


    -- if this is last batch it may not be the same size, so check that
    if image_count ~= batch_size then
        img_batch = img_batch[{{1,image_count}, {}, {}, {} } ]
    end

   -- Get the output of the layer before the (removed) fully connected layer
    local output = model:forward(img_batch:cuda()):squeeze(1)


    -- this is necesary because the model outputs different dimension based on size of input
    if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end

    if not features_test then
       features_test = torch.FloatTensor(number_of_files_test, output:size(2)):zero()
    end
       features_test[{ {i, i-1+image_count}, {}  } ]:copy(output)

    if (i - math.floor(i/500)*500 == 1) then
        print(i .. "test")
    end

end


--[[
net=loadcaffe.load(opt.cnn_proto, opt.cnn_model,opt.backend);
net:evaluate()
net=net:cuda()

function loadim(imname)
    im=image.load(imname)
    im=image.scale(im,224,224)
    if im:size(1)==1 then
        im2=torch.cat(im,im,1)
        im2=torch.cat(im2,im,1)
        im=im2
    elseif im:size(1)==4 then
        im=im[{{1,3},{},{}}]
    end
    im=im*255;
    im2=im:clone()
    im2[{{3},{},{}}]=im[{{1},{},{}}]-123.68
    im2[{{2},{},{}}]=im[{{2},{},{}}]-116.779
    im2[{{1},{},{}}]=im[{{3},{},{}}]-103.939
    return im2
end

local image_root = opt.image_root
-- open the mdf5 file

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
    table.insert(train_list, image_root .. imname)
end

local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
    table.insert(test_list, image_root .. imname)
end

local ndims=4096
local batch_size = opt.batch_size
local sz=#train_list
local feat_train=torch.CudaTensor(sz,ndims)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    ims=torch.CudaTensor(r-i+1,3,224,224)
    for j=1,r-i+1 do
        ims[j]=loadim(train_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_train[{{i,r},{}}]=net.modules[43].output:clone()
    collectgarbage()
end

print('DataLoader loading h5 file: ', 'data_train')
local sz=#test_list
local feat_test=torch.CudaTensor(sz,ndims)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    ims=torch.CudaTensor(r-i+1,3,224,224)
    for j=1,r-i+1 do
        ims[j]=loadim(test_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_test[{{i,r},{}}]=net.modules[43].output:clone()
    collectgarbage()
end
]]--

local train_h5_file = hdf5.open(opt.out_name, 'w')
--train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_train', features_train:float())
train_h5_file:write('/images_test', features_test:float())
train_h5_file:close()

