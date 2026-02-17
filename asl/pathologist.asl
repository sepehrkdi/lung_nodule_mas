// Pathologist Agent Plans (AgentSpeak for SPADE-BDI)
// Analyzes radiology reports using NLP.

// --------------------------------------------
// Initial beliefs
// --------------------------------------------
nlp_loaded(false).
nlp_model("en_core_sci_sm").

// --------------------------------------------
// Plan: Initialize NLP pipeline
// --------------------------------------------
+!initialize : nlp_loaded(false)
    <- .print("Initializing scispaCy NLP pipeline...");
       .load_nlp_model;
       -nlp_loaded(false);
       +nlp_loaded(true);
       .print("NLP pipeline ready").

// --------------------------------------------
// Plan: Analyze radiology report
// --------------------------------------------
// Main entry point for report analysis
// Calls internal actions for each extraction step
//
+!analyze(NoduleId, ReportText) : nlp_loaded(true)
    <- .print("Analyzing report for nodule: ", NoduleId);
       .extract_all(ReportText, Size, Texture, Margin, Spiculation, Assessment);
       +extraction(NoduleId, Size, Texture, Margin, Spiculation)[source(self)];
       +malignancy_assessment(NoduleId, Assessment)[source(self)];
       !send_findings(NoduleId);
       .print("Report analysis complete for: ", NoduleId).

// --------------------------------------------
// Plan: Handle analysis when NLP not loaded
// --------------------------------------------
+!analyze(NoduleId, ReportText) : nlp_loaded(false)
    <- .print("NLP not loaded, initializing first...");
       !initialize;
       !analyze(NoduleId, ReportText).

// --------------------------------------------
// Plan: Detailed extraction (alternative)
// --------------------------------------------
// More granular extraction for educational demonstration
//
+!analyze_detailed(NoduleId, ReportText) : nlp_loaded(true)
    <- .print("Detailed analysis for: ", NoduleId);
       !extract_measurements(NoduleId, ReportText);
       !extract_morphology(NoduleId, ReportText);
       !extract_impression(NoduleId, ReportText);
       !send_findings(NoduleId).

+!extract_measurements(NoduleId, Text) : true
    <- .extract_size(Text, Size);
       +size_mm(NoduleId, Size)[source(self)];
       .print("Extracted size: ", Size, " mm").

+!extract_morphology(NoduleId, Text) : true
    <- .extract_texture(Text, Texture);
       .extract_margin(Text, Margin);
       .extract_spiculation(Text, Spic);
       +texture(NoduleId, Texture)[source(self)];
       +margin(NoduleId, Margin)[source(self)];
       +spiculation(NoduleId, Spic)[source(self)].

+!extract_impression(NoduleId, Text) : true
    <- .extract_malignancy(Text, Assessment);
       .extract_lung_rads(Text, Category);
       +malignancy_text(NoduleId, Assessment)[source(self)];
       +lung_rads(NoduleId, Category)[source(self)].

// --------------------------------------------
// Plan: Send findings to oncologist
// --------------------------------------------
+!send_findings(NoduleId) : extraction(NoduleId, Size, Texture, Margin, Spic)
    <- .print("Sending NLP findings to oncologist");
       .send(oncologist, tell, pathologist_findings(NoduleId, Size, Texture, Margin, Spic)).

// Alternative when using detailed extraction
+!send_findings(NoduleId) : size_mm(NoduleId, Size) & 
                            texture(NoduleId, Texture) & 
                            margin(NoduleId, Margin)
    <- .print("Sending detailed findings to oncologist");
       ?spiculation(NoduleId, Spic);
       .send(oncologist, tell, pathologist_findings(NoduleId, Size, Texture, Margin, Spic)).

// --------------------------------------------
// Plan: Handle incoming request message
// --------------------------------------------
+request(analyze, NoduleId, ReportText)[source(Sender)]
    <- .print("Received report analysis request from: ", Sender);
       !analyze(NoduleId, ReportText).

// --------------------------------------------
// Plan: Respond to query about extraction
// --------------------------------------------
+?get_extraction(NoduleId, Result) : extraction(NoduleId, Size, Texture, Margin, _)
    <- Result = result(NoduleId, Size, Texture, Margin).

+?get_extraction(NoduleId, Result) : not extraction(NoduleId, _, _, _, _)
    <- Result = not_analyzed(NoduleId).

// --------------------------------------------
// Plan: Handle entities from NER
// --------------------------------------------
+entity(Type, Text, Start, End)[source(nlp)]
    <- .print("Found entity: ", Type, " - ", Text);
       +recognized_entity(Type, Text)[source(self)].

// ============================================
// Internal Actions (implemented in Python):
// ============================================
// .load_nlp_model            - Load scispaCy model
// .extract_all(Text, Size, Texture, Margin, Spic, Assessment) - Full extraction
// .extract_size(Text, Size)  - Extract size measurement
// .extract_texture(Text, T)  - Extract texture (solid, ground-glass, etc.)
// .extract_margin(Text, M)   - Extract margin description
// .extract_spiculation(Text, S) - Extract spiculation level
// .extract_malignancy(Text, A)  - Extract malignancy assessment
// .extract_lung_rads(Text, C)   - Extract Lung-RADS category
// ============================================
