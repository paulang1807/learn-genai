## [Gradio](https://www.gradio.app/) ([Colab](https://colab.research.google.com/drive/1dnXN35xnQbLL1ZC_rY-pigl3nqHFZNqW))
- Easily launch local web servers for your models
- Serve up react based UIs for your models
- Requires code to be implemented to be passed as a callback function
- Provides multiple options for launching the interface
    - `inbrowser=True` to launch the interface in the browser
    - `share=True` to launch a shareable link instead of a local server
        - Code is sent to Gradio in huggingface and run there
        - When user interacts with the interface, the code is accessed from the local machine
            - Uses tunneling to access the local machine
        - A more permanent solution is to use [Huggingface Spaces](https://huggingface.co/spaces)
- Provides simple mechanisms for adding authentication
    - Can pass multiple auths as a list of tuples
- Can generate output in multiple formats including markdown
    - Use `python generators` to stream output from LLM function calls ([colab example](https://colab.research.google.com/drive/1dnXN35xnQbLL1ZC_rY-pigl3nqHFZNqW#scrollTo=_kwvy8WCLgIG))

!!! example "Simple Gradio Interface"
    ```python
    import gradio as gr
    
    # Define interface for text input and text output
    # Pass function name as callback
    iface = gr.Interface(fn=myfunctionname, inputs="text", outputs="text")
    
    # Launch interface
    iface.launch()
    # Use share=True when launching from colab
    iface.launch(share=True)
    # Use inbrowser=True when running locally and want to open in browser
    iface.launch(inbrowser=True)

    # Add authentication
    iface.launch(auth=[("username1", "password1"),("username2", "password2")])

    # Add more specifications for input and output
    inputs = gr.Textbox(label="Input Label:", info=f"Input message", lines=5)
    outputs = gr.Textbox(label="Output Label:", info=f"Output message", lines=5)
    # Get output in markdown format
    outputs = gr.Markdown(label="Output Label:")

    # Add more input types
    input_dropdown = gr.Dropdown(["Option1", "Option2", "Option3"], label="Select an option", value="Option1")  # value parameter sets default value
    ```

## [Streamlit](https://streamlit.io/)
- 