import telegram


class Notifier(object):
    def __init__(self, token, chat_id, name=''):
        super(Notifier, self).__init__()
        self.bot = telegram.Bot(token=token)
        self.chat_id = chat_id
        self.name = name

    def set_token(self, token):
        self.token = token

    def set_chat_id(self, chat_id):
        self.chat_id = chat_id

    def set_name(self, name):
        self.name = name

    def notify(self, msg=None, img=None):
        assert msg is not None or img is not None
        if msg: self.bot.send_message(chat_id=self.chat_id, text='\n'.join([self.name,'==============',msg]))
        if img: self.bot.send_photo(chat_id=self.chat_id, photo=open(img, 'rb'))
