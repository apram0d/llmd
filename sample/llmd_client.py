import dbus

def main():
    bus = dbus.SessionBus()
    service = bus.get_object('com.intel.llmd', '/com/intel/llmd')

    # Get the interface
    interface = dbus.Interface(service, dbus_interface='com.intel.llmd')

    text = """Summarize the following text in two sentences: 
    Text summarization is an NLP task that creates a concise and informative summary of a longer text. LLMs can be used to create summaries of news articles, research papers, technical documents, and other types of text.

Summarizing large documents can be challenging. To create summaries, you need to apply summarization strategies to your indexed documents. You have already seen some of these strategies in the previous notebooks. If you haven't completed it, it is recommended to do so to have a basic understanding of how to summarize large documents.

In this notebook, you will use LangChain, a framework for developing LLM applications, to apply some summarization strategies. The notebook covers several examples of how to summarize large documents.
    """
    timeout_seconds = 60
    try:
        result = interface.get_dbus_method('generate', dbus_interface='com.intel.llmd')(text, timeout=timeout_seconds)
        print(result)
    except dbus.DBusException as e:
        print(f"DBus call failed: {e}")
if __name__ == '__main__':
    main()