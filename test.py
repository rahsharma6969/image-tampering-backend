from model.mantranet import pre_trained_model, check_forgery

model = pre_trained_model(weight_path='./model/MantraNetv4.pt')
check_forgery(model, img_path='./test_images/fake.jpg') 
