--
-- Krishna Murthy, Sarthak Sharma
-- Januray 2017
--


-- Load requried packages
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'

----------------------
-- Helper Functions --
----------------------

-- Recover predictions from a bunch of heatmaps
-- Takes 'hm' - a set of heatmaps - as input
function getPreds(hm)

    -- We assume the 4 heatmap dimensions are for [num images] x [num kps per image] x [height] x [width]

    -- Verify that the heatmap tensor has 4 dimensions
    assert(hm:size():size() == 4, 'Input must be 4-D tensor')
    -- Reshape the heatmap so that [height] and [width] are flattened out to a single dimension
    -- Get the maxima over the third dimension (comprising of the [height * width] flattened values)
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    -- Allocate memory for a tensor to hold the X, Y coordinates of the maxima, and the confidence score
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    -- Obtain the X coordinate of each maxima
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    -- Obtain the Y coordinate of each maxima
    preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)
    
    -- Return the predicted locations
    return preds

end


-- A function to perform slight post-processing, to be accurate to the pixel level
function postprocess(output, outputRes)
    
    -- Obtain keypoint predictions from the output heatmaps
    local p = getPreds(output)
    -- Initialize a tensor to hold the prediction confidences (pixel intensities at the output location)
    local scores = torch.zeros(p:size(1),p:size(2),1)

    -- Very simple post-processing step to improve performance at tight PCK thresholds
    for i = 1, p:size(1) do
        for j = 1, p:size(2) do
            local hm = output[i][j]
            local pX, pY = p[i][j][1], p[i][j][2]
            scores[i][j] = hm[pY][pX]
            if pX > 1 and pX < outputRes and pY > 1 and pY < outputRes then
               local diff = torch.Tensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(0.5)
   
    return p:cat(p,3):cat(scores,3)

end


-----------------
-- Main Script --
-----------------

-- Set the default tensor type to a FloatTensor
torch.setdefaulttensortype('torch.FloatTensor')

-- Number of keypoints (set according to the object category)
numKps = 16

-- Number of hourglass modules stacked
numStack = 2


-- Dimensions of each prediction
predDim = {numKps, 5}
-- Dimension of each input to the network
inputDim = {3, 64, 64}
-- Dimension of each output from the newtork
outputDim = {}
for i = 1,numStack do
    outputDim[i] = {numKps, 64, 64}
end
-- Resolution of the output image (assumed to be square)
outputRes = 64


-- Input file (contains image paths and bboxes)
-- Syntax of each line: /full/path/to/image x y w h
-- Here, the image refers to the entire image (eg. a KITTI frame)
-- x, y, w, h are "0-based" indices of a car bounding box
FolderPath = '/home/dinesh/CarCrash/data/Fifth/' 
FolderPath = '/home/dinesh/CarCrash/data/CarCrash/Cleaned/' 

-- Path to the saved model (.t7 file)
modelPath_car = '/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/trained_models/car_model.t7'

modelPath_person = '/home/dinesh/CarCrash/codes/CollisionRecon/keypoints/trained_models/person_model.t7'

-- Path to the results file, where keypoint predictions will be written
resultPath = 'results.txt'

-- Load the model
print('Loading the model ...')
if nn.DataParallelTable then
	-- set number of GPUs to use when deserializing model
	nn.DataParallelTable.deserializeNGPUs = 1
end
model_car = torch.load(modelPath_car)
model_car:cuda()
model_person = torch.load(modelPath_person)
model_person:cuda()

print('Predicting keypoints')

for hh = 2,26 do
	for dataPath in io.popen('ls ' .. FolderPath .. tostring(hh-1) .. '/boundingbox/*.txt'):lines() do

		-- Determine the number of images
		local numImgs = 0;
		for line in io.lines(dataPath) do 
			numImgs = numImgs + 1;
		end

		-- Initialize variables to save the images, predictions, and their heatmaps
		saved = {idxs = torch.Tensor(numImgs), preds = torch.Tensor(numImgs, unpack(predDim))}
		-- saved.input = torch.Tensor(numImgs, unpack(inputDim))
		-- saved.heatmaps = torch.Tensor(numImgs, unpack(outputDim[1]))



		-- For each instance whose kps are to be predicted
		i = 1;
		keypoints = {}
		bb = {}
		filename = dataPath:split("/")[table.getn(dataPath:split("/"))]
		cimgpath = FolderPath .. tostring(hh-1) ..'/' .. tostring(filename:split(".txt")[1])
		resultPath = FolderPath .. tostring(hh-1) ..'/keypoints_txt/' .. tostring(filename)
		print(resultPath)
		--resultPath = dataPath:replace('boundingbox','sds')
		--print(table.getn(dataPath:split("/")))
		for line in io.lines(dataPath) do
			-- Load the image from a text file,format : /path/to/text/file x y w h
			cx, cy, cx2, cy2, class = unpack(line:split(" "));
			-- Image path
			cimg = image.load(cimgpath)
			-- (0-based) X coordinate of top left corner of bbox
			cx = tonumber(cx);
			-- (0-based) Y coordinate of top left corner of bbox
			cy = tonumber(cy);
			-- Width of bbox
			cw = tonumber(cx2) - cx;
			-- Height of bbox
			ch = tonumber(cy2) - cy;
		   
			-- Converting the image to a float tensor (by default, images are loaded as userdata, not tensors)
			cimg = torch.FloatTensor(cimg);
			-- Cropping the car according to the specified bbox
			-- Adding 1 to account for the fact that Torch indexing is 1-based
			-- Also, note that we're not doing cx+cw-1 (since we're using a 1-based index)
			carImg = image.crop(cimg, cx+1, cy+1, cx+cw, cy+ch)
			-- Scaling the image to the input resolution

			
			-- Getting output from the network
			if class == 'car' or class == 'truck' then
				scImg = image.scale(carImg, 64, 64)
				
				-- Creating the input tensor
				input = torch.Tensor(1, 3, 64, 64);
				input[1] = scImg;
				output_v = model_car:forward(input:cuda())
			end
			if class == 'person' then 
				scImg = image.scale(carImg, 256, 256)
				
				-- Creating the input tensor
				input = torch.Tensor(1, 3, 256, 256);
				input[1] = scImg;
				output_v = model_person:forward(input:cuda())
			end
			-- output_v = model_person:forward(input:cuda())
			-- Output is a table of 16 heatmaps, two from each hourglass. The 15th entry is what we need.
			-- The other entries correspond to predictions that are either from a lower layer, or predictions 
			-- that are not necessary.
			if type(output_v) == 'table' then
				output = output_v[#output_v]
			end

			-- Saving the predictions for the image
			-- saved.input[i]:copy(input[1])
			
			-- Obtain the keypoints from the output heatmaps from the network
			keyPoints = postprocess(output, outputRes);

			-- Copy them to the 'saved' tensor, to write them to an output file
			-- saved.preds[i]:copy(keyPoints[1])
			table.insert(keypoints,keyPoints[1])
			table.insert(bb,{cx,cy,cw,ch,class})
			-- Increment the index
			i = i + 1;
		end

		-- Write the predictions to the output text file
		fd = io.open(resultPath, 'w')
		for i = 1, numImgs do
			-- Write the keypoint X and Y coordinates (1-based) and the confidence scores (comma-separated)
			fd:write(tostring(i) ..','.. tostring(bb[i][1])..','..tostring(bb[i][2])..','..tostring(bb[i][3])..','..tostring(bb[i][4])..',')
			for j = 1,keypoints[i]:size(1) do
				fd:write(tostring(keypoints[i][j][1])..','..tostring(keypoints[i][j][2])..','..tostring(keypoints[i][j][5]))
				if j ~= keypoints[i]:size(1) then
					fd:write(tostring(','))
				end
			end
			fd:write(',' .. tostring(bb[i][5]))
			fd:write(tostring('\n'))
		end
	end
end
