from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import numpy as np
import pickle
animal_num_epoch = 1250
epoch_num = int(12500/animal_num_epoch)

def image_preprocess(animal = "cat",
					 animal_num_epoch = animal_num_epoch,
					 epoch_num = epoch_num):
	for j in range(epoch_num):
		X_train = np.zeros((animal_num_epoch, 224, 224, 3))
		if animal == "cat":
			num = 0
			y_train = np.zeros((animal_num_epoch, 1)).astype(int)
		else:
			num = 10
			y_train = np.ones((animal_num_epoch, 1)).astype(int)
		for i in range(j*animal_num_epoch, (j+1)*animal_num_epoch):
			img_path = 'train/'+str(animal)+'.'+str(i)+'.jpg'
			img = image.load_img(img_path, target_size = (224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis = 0)
			x = preprocess_input(x)
			X_train[i-j*animal_num_epoch] = x
			#print(x.shape)
			#print("*******************************************************")
		pickle.dump((X_train, y_train), open("train_image_"+str(j+num), 'wb'))
		print(str(animal)+" train_image_"+str(j)+" saved")


if __name__ == "__main__":
	image_preprocess(animal = 'cat')
	image_preprocess(animal = 'dog')


from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import numpy as np
import pickle
def test_process():

	for j in range(0, 10):
		X_test = np.zeros((1250, 224, 224, 3))
		for i in range((int)(j*1250)+1, (int)((j+1)*1250)+1):
			img_path = 'test/'+str(i)+'.jpg'
			img = image.load_img(img_path, target_size = (224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis = 0)
			x = preprocess_input(x)
			X_test[i-((int)(j*1250)+1)] = x
		pickle.dump(X_test, open("test_image_"+str(j), 'wb'))
		print("test set propress batch:"+ str(j))
	print("test set prepress OK!")


if __name__ == "__main__":
	test_process()
