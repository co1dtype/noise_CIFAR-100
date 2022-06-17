def model_save(model_num = ""):
    PATH = './'

    torch.save(model, PATH + 'model' + model_num + '.pt')  # 전체 모델 저장
    torch.save(model.state_dict(), PATH + 'model' + model_num + '_state_dict.pt')  # 모델 객체의 state_dict 저장
    torch.save({
        'model': model.state_dict(),
        'optimizer': opti.state_dict()
    }, PATH + 'all' + model_num + '.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능
