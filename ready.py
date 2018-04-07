import shelve
import Myclass as mc
import MySetting as ms

gigaVocab = mc.GigaLoader(0, 50000)

lang = mc.Lang('lang')
lang.index_giga(gigaVocab)
obj_file = shelve.open(ms.shelve_Path+'myshelve')
obj_file['lang'] = lang
obj_file.close()

vocab_size = lang.n_words

wv = mc.wordVector(ms.wv_Path, lang.word2index, size=30000)
enco = mc.Encoder(vocab_size, ms.h_size, wv.veclist, n_layers=2)
deco = mc.AttnDecoder(hidden_size=ms.h_size, output_size=vocab_size, n_layers=2)
enco.save_para(ms.net_Path, "mini_Enco.pkl")
deco.save_para(ms.net_Path, "mini_Deco.pkl")
