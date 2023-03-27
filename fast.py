import fasttext
import fasttext.util
ft = fasttext.load_model('cc.en.300.bin')
ft.get_dimension()
fasttext.util.reduce_model(ft, 100)
ft.get_dimension()
ft.save_model('cc.en.100.bin')