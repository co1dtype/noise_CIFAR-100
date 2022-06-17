def evaluate(model, dataloader, criterion):
    scaler = torch.cuda.amp.GradScaler()
    correct = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(dataloader, total=len(dataloader)):
            data = data.to('cuda')
            target = target.to('cuda')


            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item()
    
    acc = 100. * correct / len(dataloader.dataset)
    total_loss /= len(dataloader.dataset)
    return acc, total_loss
