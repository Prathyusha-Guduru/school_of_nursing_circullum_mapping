import gradio as gr
import json

from pdf_to_json import (
    extract_syllabi_from_uploaded,
    extract_aacn_from_uploaded
)

from method_1_bert import run_bert_topic


# --------------------------------------------------
# MAIN PIPELINE FUNCTION
# --------------------------------------------------

def run_full_pipeline(mode, syllabi_files, aacn_files, syllabi_json_file, aacn_json_file, threshold):

    # ---------- ALWAYS RETURN 8 VALUES ----------
    def out(
        fig_topics=None,
        fig_hierarchy=None,
        fig_heatmap=None,
        coverage_md=None,
        match_df=None,
        matrix_df=None,
        sankey_fig=None,
        status="",
    ):
        return (
            fig_topics,
            fig_hierarchy,
            fig_heatmap,
            coverage_md,
            match_df,
            matrix_df,
            sankey_fig,
            status,
        )

    # INITIAL STATUS
    yield out(status="Starting pipeline...")

    # --------------------------------------------------
    # MODE: USE PDF FILES
    # --------------------------------------------------
    if mode == "Use PDF Files":

        if not syllabi_files:
            yield out(status="Please upload syllabus PDFs.")
            return

        if not aacn_files:
            yield out(status="Please upload AACN PDFs.")
            return

        yield out(status="Extracting syllabi PDFs...")
        syllabi = extract_syllabi_from_uploaded(syllabi_files)
        yield out(status=f"Extracted {len(syllabi)} syllabi documents.")

        yield out(status="Extracting AACN PDFs...")
        aacn = extract_aacn_from_uploaded(aacn_files)
        yield out(status=f"Extracted {len(aacn)} AACN domains.")

    # --------------------------------------------------
    # MODE: USE JSON FILES
    # --------------------------------------------------
    elif mode == "Use Pre-Existing JSON":

        if syllabi_json_file is None:
            yield out(status="Upload syllabus JSON.")
            return

        if aacn_json_file is None:
            yield out(status="Upload AACN JSON.")
            return

        yield out(status="Loading JSON files...")
        syllabi = json.load(open(syllabi_json_file.name, "r"))
        aacn = json.load(open(aacn_json_file.name, "r"))
        yield out(status="Loaded structured JSON files.")

    else:
        yield out(status="Unknown mode selected.")
        return


    # --------------------------------------------------
    # BUILD course + domain data for topic modeling
    # --------------------------------------------------
    yield out(status="Preparing documents for topic modeling...")

    # Build aacn_domains dict
    aacn_domains = {
        d["domain_name"]: d.get("progression_indicators", [])
        for d in aacn
        if d.get("domain_name")
    }

    # Build combined documents for BERTopic
    all_docs = []
    for s in syllabi:
        all_docs += s.get("learning_objectives", [])
        all_docs += s.get("learning_outcomes", [])
        all_docs += s.get("topical_outline", [])

    for pis in aacn_domains.values():
        all_docs += pis

    if len(all_docs) == 0:
        yield out(status="No textual content found.")
        return

    yield out(status=f"Collected {len(all_docs)} text entries.")


    # --------------------------------------------------
    # RUN BERT TOPIC + SIMILARITY + SANKEY
    # --------------------------------------------------
    yield out(status="Running BERT Topic (this may take a while)...")

    (
        fig_topics,
        fig_hierarchy,
        fig_heatmap,
        coverage_report,
        match_df,
        matrix_df,
        sankey_fig,
    ) = run_bert_topic(all_docs, syllabi, aacn_domains, threshold=0.8)


    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------
    yield out(
        fig_topics=fig_topics,
        fig_hierarchy=fig_hierarchy,
        fig_heatmap=fig_heatmap,
        coverage_md=coverage_report,
        match_df=match_df,
        matrix_df=matrix_df,
        sankey_fig=sankey_fig,
        status="Completed BERT Topic."
    )

# --------------------------------------------------
# GRADIO UI
# --------------------------------------------------

with gr.Blocks() as demo:

    gr.Markdown("## 🧠 Curriculum Mapper")
    gr.Markdown("Choose whether to upload PDFs or pre-extracted JSON files.")

    # MODE SELECTOR
    mode_selector = gr.Radio(
        ["Use PDF Files", "Use Pre-Existing JSON"],
        label="Choose Input Mode",
        value="Use PDF Files"
    )

    # PDF INPUTS
    with gr.Row(visible=True) as pdf_row:
        syllabi_input = gr.File(label="Upload syllabi PDFs", file_types=[".pdf"], file_count="multiple")
        aacn_input = gr.File(label="Upload AACN PDFs", file_types=[".pdf"], file_count="multiple")

    # JSON INPUTS
    with gr.Row(visible=False) as json_row:
        syllabi_json_input = gr.File(label="Upload syllabi_extracted.json", file_types=[".json"])
        aacn_json_input = gr.File(label="Upload aacn_domain_consolidated.json", file_types=[".json"])

    # MODE SWITCH VISIBILITY
    def update_visibility(mode):
        return (
            gr.update(visible=(mode == "Use PDF Files")),
            gr.update(visible=(mode == "Use Pre-Existing JSON"))
        )

    mode_selector.change(
        update_visibility,
        inputs=mode_selector,
        outputs=[pdf_row, json_row]
    )

    threshold_input = gr.Slider(
    minimum=0.0,
    maximum=1.0,
    value=0.50,
    step=0.05,
    label="Similarity Threshold (Course → Domain Match)"
    )

    run_button = gr.Button("Run Analysis", variant="primary")

    # ------- OUTPUT TABS -------
    with gr.Tabs():

        with gr.Tab("Topic Visualizations"):
            out1 = gr.Plot(label="Topics Overview")
            out2 = gr.Plot(label="Hierarchy")
            out3 = gr.Plot(label="Heatmap")

        with gr.Tab("Domain Coverage Report"):
            coverage_md = gr.Markdown()

        with gr.Tab("Top Matches (Course → Domain)"):
            match_table_df = gr.DataFrame()

        with gr.Tab("Course × Domain Similarity Matrix"):
            matrix_df = gr.DataFrame()

        with gr.Tab("Sankey Diagram"):
            sankey_plot = gr.Plot(label="Courses in each Domain")

    status = gr.Textbox(label="Status", lines=3)

    run_button.click(
        fn=run_full_pipeline,
        inputs=[
            mode_selector,
            syllabi_input,
            aacn_input,
            syllabi_json_input,
            aacn_json_input,
            threshold_input
        ],
        outputs=[
            out1, out2, out3,
            coverage_md,
            match_table_df,
            matrix_df,
            sankey_plot,
            status
        ]
    )

demo.launch()

