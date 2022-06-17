def test(model, dataloader):
    scaler = torch.cuda.amp.GradScaler()
    y_pred = []
    
    for data, _ in dataloader:
        data = data.to('cuda')
    
        with torch.cuda.amp.autocast():
            output = model(data)
        pred = output.max(1, keepdim=True)[1]
        y_pred += pred.tolist()
    
    y_pred = np.array(y_pred).reshape(-1, )
    return y_pred
