def train(model, dataloader, criterion, opti):
    scaler = torch.cuda.amp.GradScaler()
    correct = 0
    total_loss = 0
    
    model.train()
    for data, target in tqdm(dataloader, total=len(dataloader)):
        data = data.to('cuda')
        target = target.to('cuda')
        
        opti.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(opti)
        scaler.update()
        
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item()
    
    acc = 100. * correct / len(dataloader.dataset)
    total_loss /= len(dataloader.dataset)
    return acc, total_loss
