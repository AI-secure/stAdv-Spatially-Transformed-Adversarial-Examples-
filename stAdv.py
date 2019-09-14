
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import numpy as np 


class stAdvAttack:
    def __init__(self, model, args):

        self.model = model 
        self.args = args 

        super(stAdvAttack, self).__init__()

    def __generate_img_from_flow(self, flow, grid, image):
        flow2 = flow.transpose(1,2).transpose(2,3)
        grid2 = grid + flow2
        grid3 = grid2.clamp(-1, 1)
        fake = F.grid_sample(image, grid3)
        loss_tv = torch.sqrt(torch.mean((flow2[:,1:,:,:] - flow2[:,:-1,:,:])**2)\
                                + torch.mean((flow2[:,:,1:,:] - flow2[:,:,:-1,:])**2)  + 10e-10)
        # loss_tv = torch.mean((flow2[:,1:,:,:] - flow2[:,:-1,:,:])**2)\
                                    # + torch.mean((flow2[:,:,1:,:] - flow2[:,:,:-1,:])**2)
    
        return fake, loss_tv

    def __adv_loss(self, model, target_labels, adv_images):
        predict_labels = model(adv_images)
        real = torch.sum( predict_labels * target_labels, 1)
        # pdb.set_trace()
        other, _ = torch.max( ( 1 - target_labels) * predict_labels - target_labels * 1000, 1)
        # pdb.set_trace()
        loss = torch.clamp( other - real + 0.1, min = 0)
        loss = torch.mean(loss)    
        return loss

    def perturb(self, X, y):
        b, n, w, h = X.size()

        # create a grid 
        M = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0] * b, dtype=np.float32)
        M_tensor = torch.from_numpy(M).view(b, 2, 3).to(X.device)
        grid = F.affine_grid(M_tensor, X.size())

        self.grid = grid
        flow = torch.zeros(b, 2, w, h ).to(X.device)
        flow.requires_grad = True
        optimizer = optim.LBFGSB([flow], lr = 0.03,  max_iter = 500, line_search_fn = 'backtracking')
        success_flow = None
        big_loss_tv = 1e10
        step = 0
    
        def closure():
            nonlocal step
            nonlocal big_loss_tv
            nonlocal success_flow
            optimizer.zero_grad()
            adv_images, loss_tv = self.__generate_img_from_flow(flow, grid, X)

            loss_adv = self.__adv_loss(self.model, y, adv_images) 

            if self.args.hingle_flag:
                loss = self.args.ld_tv * loss_tv + self.args.ld_adv * torch.clamp(  loss_adv , min = 0)
            else:
                loss = self.args.ld_tv * loss_tv + self.args.ld_adv * loss_adv
            loss.backward()

            if loss_adv.item() < 1e-3 and loss_adv.item() < big_loss_tv:
                success_flow = flow
                big_loss_tv = loss_adv.item()     
            # print(step, loss.item(), loss_adv.item(), loss_tv.item())
            step += 1
            return loss
        optimizer.step(closure)
        if big_loss_tv != 1e10:
            adv_images, loss_tv = self.__generate_img_from_flow(success_flow, grid, X)
        else:
            adv_images, loss_tv = self.__generate_img_from_flow(flow, grid, X)
        # logit = self.model(adv_images)
        # _, pred = torch.max(logit, 1)
        return adv_images


def image_loader(image_name, loader):
    image = Image.open(image_name)
    image = Variable(loader(image).cuda(), requires_grad = False)
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


if __name__ == "__main__":
    import torchvision
    from torchvision import models, transforms
    import os 
    from opt_imagenet import parse_opt
    import csv
    import requests
    from PIL import Image
    from pdb import set_trace as st
    opt = parse_opt()   
        
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = torch.cuda.is_available()
    model = models.inception_v3(pretrained=True).cuda()
    model.eval()   
    attack = stAdvAttack(model=model, args=opt) 
    LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    
    opt.fineSize = 299
    preprocess = transforms.Compose([
        transforms.Scale((299,299)),
        transforms.ToTensor(),
        normalize
    ])
    input_filename = opt.image_path + 'dev_dataset.csv'

    with open(input_filename) as input_file:
        reader = csv.reader(input_file)
        header_row = next(reader)
        rows = list(reader)
    try:
        row_idx_image_id = header_row.index('ImageId')
        row_idx_slb = header_row.index('TrueLabel')
        row_idx_tlb = header_row.index('TargetClass')

    except ValueError as e:
        print('One of the columns was not found in the source file: ',
              e.message)

    rows = [(row[row_idx_image_id], int(row[row_idx_slb])-1 , int(row[row_idx_tlb])-1  )for row in rows]
    classes = {int(key):value for (key, value) in requests.get(LABELS_URL).json().items()}
    num_classes = 1000
    s_count = 0
    count = 0
    for i in range(0, 1):
        row = rows[i]
        full_path = "{}/images/{}.png".format(opt.image_path, row[0])
        print (full_path)
        img_variable = image_loader(full_path, preprocess)
        # image1 = imread(full_path, mode='RGB').astype(np.float)
        # st()
        # img_tensor = preprocess(image1)
        # imgf_variable = Variable(img_tensor.unsqueeze(0).cuda(), requires_grad = False)
        logit = model(img_variable)
        h_x = F.softmax(logit).data.squeeze()
        probs, idx = h_x.sort(0, True)
        inputs = img_variable
        original_idx = idx
        # output the predictio
        for i in range(0, 5):
            print('{:.3f} -> {} : {}'.format(probs[i], classes[idx[i].item()], classes[row[1]]))


        t_l = int(row[2])
        target_labels = np.zeros([1, num_classes], dtype = np.float32)
        target_labels[:, t_l] = 1
        target_labels = torch.from_numpy(target_labels).cuda()

        adv_inputs = attack.perturb(inputs, target_labels)
        logits = model(adv_inputs)
        _, preds = torch.max(logits,1)
        
        if preds.item() == t_l:
            s_count += 1
        count += 1 
        print("{}/{}".format(s_count, count))
