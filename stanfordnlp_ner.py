''' 创建stanfordnlp服务，并且对文本计算ner结果 '''


from stanfordcorenlp import StanfordCoreNLP
import settings


# class StanfordNER():
#     ''' 调用stanfordnlp工具做ner
#     '''
#     def __init__(self, text):
#         # 创建stanfordnlp工具，做ner
#         # nlp = StanfordCoreNLP(r'D:\stanford-corenlp-4.5.9',
#         #                       lang="zh",
#         #                       )
#         # nlp = StanfordCoreNLP(r'D:\stanford-corenlp-4.5.9', lang='zh')
#         nlp = StanfordCoreNLP(
#             r'D:\stanford-corenlp-4.5.9',
#             lang='zh',
#             model_path=r'D:\stanford-corenlp-4.5.9\stanford-chinese-corenlp-4.5.9-models.jar')
#         self.ner_result = nlp.ner(text)
#         nlp.close()#运行结束关闭模型否则占用大量内存

import stanza

import stanza

class StanfordNER():
    def __init__(self, text):
        nlp = stanza.Pipeline(
            'zh',
            processors='tokenize,ner',
            use_gpu=False,
            download_method=stanza.DownloadMethod.REUSE_RESOURCES  # ✅ 关键是这行
        )
        doc = nlp(text)
        self.ner_result = [(ent.text, ent.type) for ent in doc.entities]

# class StanfordNER():
#     def __init__(self, text):
#         # 初始化管道（只需下载一次模型，建议只在首次单独运行 stanza.download）
#         # stanza.download('zh')  # 可注释掉，首次手动执行即可
#         nlp = stanza.Pipeline('zh', processors='tokenize,ner', use_gpu=False)
#         doc = nlp(text)
#         self.ner_result = [(ent.text, ent.type) for ent in doc.entities]


if __name__ == "__main__":
    pass



