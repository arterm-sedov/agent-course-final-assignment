with gr.Blocks() as demo:
    gr.Markdown("# GAIA Unit 4 Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit" button, it can take quite some time (this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a separate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    init_log_file = gr.File(label="Download LLM Initialization Log")
    results_log_file = gr.File(label="Download Full Results Log")
    results_csv_file = gr.File(label="Download Results Table (CSV)")
    score_file = gr.File(label="Download Final Score/Status")
    file_state = gr.State([None, None, None, None])  # [init_log, results_log, csv, score]

    # On app load, show the latest logs (if available), using state if present
    demo.load(
        fn=get_latest_logs,
        inputs=[file_state],
        outputs=[init_log_file, results_log_file, results_csv_file, score_file, file_state],
    )

    # The run_and_submit_all function expects a profile as the first argument, which Gradio will provide after login
    run_button.click(
        fn=run_and_submit_all,
        inputs=[gr.OAuthProfile(), file_state],
        outputs=[status_output, results_table, init_log_file, results_log_file, results_csv_file, score_file, file_state]
    ) 