import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import textwrap
import mistletoe
import pandas as pd
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from google.colab import userdata

def df_plotting(x,y,x_label,y_label,title,plot_type,color):
    color="blue"
    if plot_type == "histogram":
      plt.hist(x, bins=3,color=color, edgecolor='black')
    elif plot_type == "piechart":
      plt.pie(x, labels=y, autopct='%1.1f%%', startangle=90)
    elif plot_type == "bar":
      plt.bar(y,x, color=color, width=0.5)
    elif plot_type == "linechart" or plot_type == "line":
      plt.plot(x, y, label='Cosine Curve',color=color, linestyle='-', linewidth=2, marker='o', markersize=5)
    elif plot_type == "scatterplot":
      plt.scatter(x, y, color=color, marker='o', label='data point')
    elif plot_type == "simplegraph":
      plt.plot(x, y,color=color)
    elif plot_type == "boxplot":
      plt.boxplot(df1['salary','id'], vert=True, patch_artist=True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    st.pyplot()
    plt.show()
def plot_out(output):
  otpl=output.split(",")
  a=(((otpl[0].split())[-1]).lower())
  a=a[1:-1]
  b=(((otpl[1].split())[-1]).lower())
  b=b[1:-1]
  c=(((otpl[2].split())[-1]).lower())
  d=(((otpl[3].split())[-1]).lower())
  earr=(((otpl[4].split())))
  e=" "
  for i in range (1,len(earr)):
    e=e+earr[i]+" "
  f=(((otpl[5].split())[-1]).lower())
  f=f[1:-1]
  g=(((otpl[6].split())[-1]).lower())
  g=g[1:-1]
  df_plotting(df1[a],df1[b],c,d,e,f,g)
def load_llm():
    llm = CTransformers(
        model = "/content/drive/MyDrive/main project/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens = 550,
        temperature = 0.9
    )
    return llm

st.title("IDEN")

csv_data = st.sidebar.file_uploader("Upload your Data", type="csv")

if csv_data :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(csv_data.getvalue())
        tmp_file_path = tmp_file.name
    df1=pd.read_csv(tmp_file_path)
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name='thenlper/gte-large',
                                       model_kwargs={'device': 'cpu'})
    st.write("Please wait for a time, your data is in processing stage")

    db = FAISS.from_documents(data, embeddings)
    db.save_local('faiss/ex')
    llm = load_llm()

    def to_markdown(text):
      text = text.replace('•', '  *')
      return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
    GOOGLE_API_KEY="AIzaSyAUWjOTM72flchdhS9tmu_1ZlMi0wE0xIk"
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    def df_to_txt(df):
      cnl = list(df.columns.values.tolist())
      print(cnl)
      df_tx=" "
      for i in range(len(cnl)):
        df_tx=df_tx+cnl[i]
        df_tx=df_tx+":["
        for j in range(len(df)):
          if (j <= (len(df))-2):
            df_tx=df_tx+str(df[cnl[i]][j])+","
          else:
            df_tx=df_tx+str(df[cnl[i]][j])+"],"
      return(df_tx)
    df_tx=df_to_txt(df1)
    ads1="give x_axis array name,y_axis array name,x_label,y_label,title,plot_type,color for the graph in an array like x_axis : value within quotes and with coma , y_axis : value within quotes and with coma, x_label : value within quotes and with coma,y_label : value within quotes and with coma,title : value within quotes and with coma,plot_type : value within quotes and with coma,color : value within quotes and with coma, etc"

    prompt_temp = '''
With the information provided try to answer the question.
If you cant answer the question based on the information either say you cant find an answer or unable to find an answer. So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

Context: {context}
Question: {question}
Do provide only correct answers

Correct answer:
    '''
    custom_prompt_temp = PromptTemplate(template=prompt_temp,
                            input_variables=['context', 'question'])

    retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        retriever=db.as_retriever(search_kwargs={'k': 1}),
                                        chain_type="stuff",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": custom_prompt_temp}
                                    )

    def IDEN(query):
        answer = retrieval_qa_chain({"query": query})
        return answer["result"]

    if 'user' not in st.session_state:
        st.session_state['user'] = ["Hey there"]

    if 'assistant' not in st.session_state:
        st.session_state['assistant'] = ["Hello I am IDEN and I am a new emmerging chatbot which was created by 3 college students Kailash.M,Praveen.S and Dinesh Babu.K.\nThe word IDEN stands for Interactive Data Exploration With NLP.\nIf you upload your csv file and ask question from it, I will provide answer to you.\nEven though I'm a text based bot i can do some basic visualization like bar char,pie chart etc" ]

    container = st.container()

    with container:
        with st.form(key='eda_form', clear_on_submit=True):

            user_input = st.text_input("", placeholder="Place Your Query here", key='input')
            submit = st.form_submit_button(label='Kick')

        if submit:
          st.session_state['user'].append(user_input)
          cmp=["histogram","piechart","barchart","linechart","scatterplot","simplegraph","boxplot","chart","graph","visualise"]
          ads="give x_axis array name,y_axis array name,x_label,y_label,title,plot_type,color for the graph in an array like x_axis : value within quotes and with coma , y_axis : value within quotes and with coma, x_label : value within quotes and with coma,y_label : value within quotes and with coma,title : value within quotes and with coma,plot_type : value within quotes and with coma,color : value within quotes and with coma, etc"
          user_input=user_input.lower()
          quest=user_input.split()
          res="no"
          for i in range(len(cmp)):
            if cmp[i] in quest:
              res="yes"
          if res=="yes":
            user_input=ads + user_input
            output = IDEN(user_input)
            st.session_state['assistant'].append(output)
            plot_out(output)

          else:
            #output = IDEN(user_input)
            response = model.generate_content("From "+df_tx+user_input)
            output = mistletoe.markdown(response.text)
            st.session_state['assistant'].append(output)


    if st.session_state['assistant']:
        for i in range(len(st.session_state['assistant'])):
            message(st.session_state["user"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["assistant"][i], key=str(i))
