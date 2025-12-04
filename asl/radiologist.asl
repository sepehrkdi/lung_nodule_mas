// ============================================
// Radiologist Agent Plans (AgentSpeak for SPADE-BDI)
// ============================================
//
// EDUCATIONAL PURPOSE - AGENTSPEAK SYNTAX:
//
// AgentSpeak is a logic-based agent programming language.
// Plans have the form:
//   +trigger : context <- body.
//
// Triggers:
//   +belief     - belief addition
//   -belief     - belief deletion
//   +!goal      - achievement goal adoption
//   +?goal      - test goal
//
// Body actions:
//   .internal_action(args)  - Python function call
//   !subgoal                - adopt subgoal
//   ?query                  - belief query
//   +belief                 - add belief
//   -belief                 - remove belief
//
// RADIOLOGIST ROLE:
// Analyzes CT images using DenseNet121 CNN.
// Internal actions call Python functions for ML inference.
// ============================================

// --------------------------------------------
// Initial beliefs
// --------------------------------------------
model_loaded(false).
model_name("DenseNet121").
threshold(suspicious, 0.5).

// --------------------------------------------
// Plan: Initialize classifier on startup
// --------------------------------------------
+!initialize : model_loaded(false)
    <- .print("Initializing DenseNet121 classifier...");
       .load_classifier;
       -model_loaded(false);
       +model_loaded(true);
       .print("Classifier ready").

// --------------------------------------------
// Plan: Analyze nodule image
// --------------------------------------------
// Triggered when receiving an analysis request
// Context: Model must be loaded
// Body: Run classification and extract features
//
+!analyze(NoduleId, ImageData) : model_loaded(true)
    <- .print("Analyzing nodule: ", NoduleId);
       .classify_image(NoduleId, ImageData, Probability, PredictedClass);
       +classification(NoduleId, Probability, PredictedClass)[source(self)];
       .extract_features(NoduleId, ImageData, Size, Texture, Shape);
       +visual_features(NoduleId, Size, Texture, Shape)[source(self)];
       !send_findings(NoduleId);
       .print("Analysis complete for: ", NoduleId).

// --------------------------------------------
// Plan: Handle analysis when model not loaded
// --------------------------------------------
+!analyze(NoduleId, ImageData) : model_loaded(false)
    <- .print("Model not loaded, initializing first...");
       !initialize;
       !analyze(NoduleId, ImageData).

// --------------------------------------------
// Plan: Send findings to oncologist
// --------------------------------------------
+!send_findings(NoduleId) : classification(NoduleId, Prob, Class) & 
                            visual_features(NoduleId, Size, _, _)
    <- .print("Sending findings to oncologist");
       .send(oncologist, tell, radiologist_findings(NoduleId, Prob, Class, Size)).

// --------------------------------------------
// Plan: Respond to query about classification
// --------------------------------------------
+?get_classification(NoduleId, Result) : classification(NoduleId, Prob, Class)
    <- Result = result(NoduleId, Prob, Class).

+?get_classification(NoduleId, Result) : not classification(NoduleId, _, _)
    <- Result = not_analyzed(NoduleId).

// --------------------------------------------
// Plan: Handle incoming request message
// --------------------------------------------
+request(analyze, NoduleId, ImageData)[source(Sender)]
    <- .print("Received request from: ", Sender);
       !analyze(NoduleId, ImageData).

// --------------------------------------------
// Plan: Re-evaluate when threshold changes
// --------------------------------------------
+threshold(suspicious, NewThreshold)
    <- .print("Threshold updated to: ", NewThreshold);
       !reevaluate_pending.

+!reevaluate_pending : true
    <- .findall(N, classification(N, _, _), Nodules);
       !reevaluate_list(Nodules).

+!reevaluate_list([]) <- true.
+!reevaluate_list([H|T])
    <- !reevaluate(H);
       !reevaluate_list(T).

+!reevaluate(NoduleId) : classification(NoduleId, Prob, _) & threshold(suspicious, Th)
    <- .print("Re-evaluating: ", NoduleId);
       NewClass = if(Prob >= Th, suspicious, benign);
       -classification(NoduleId, Prob, _);
       +classification(NoduleId, Prob, NewClass).

// ============================================
// Internal Actions (implemented in Python):
// ============================================
// .load_classifier          - Load DenseNet121 model
// .classify_image(Id, Img, Prob, Class) - Run classification
// .extract_features(Id, Img, Size, Texture, Shape) - Extract visual features
// ============================================
