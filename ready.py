import shelve
import Myclass as mc
import MySetting as ms

gigaVocab = mc.GigaLoader(0, 5089620)

lang = mc.Lang('lang')
lang.index_giga(gigaVocab)
obj_file = shelve.open(ms.shelve_Path+'myshelve')
obj_file['lang'] = lang
obj_file.close()

vocab_size = lang.n_words

wv = mc.wordVector(ms.wv_Path, lang.word2index, size=5089620)
enco = mc.Encoder(vocab_size, ms.h_size, wv.veclist, n_layers=1)
deco = mc.AttnDecoder(hidden_size=ms.h_size, output_size=vocab_size, n_layers=1)
enco.save_para(ms.net_Path, "mini_Enco.pkl")
deco.save_para(ms.net_Path, "mini_Deco.pkl")

