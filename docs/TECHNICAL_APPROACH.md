# Technical Approach

## 1. Problem interpretation

The business requirement is to create a reliable measurement representation for a downstream material-calculation engine. A wrong or silently missed measurement can propagate into incorrect quantities and cost calculations.

Therefore, the design must preserve both:

1. extracted values, and
2. evidence / uncertainty about how those values were obtained.

The prototype uses a hybrid approach:

- OpenCV preprocessing
- EasyOCR text recognition
- deterministic parsing
- spatial heuristics
- confidence aggregation
- validation against ground truth

The first version focuses on image-based plans and explicit dimensions.

## unit assumption 

The sample plans has dimensions without units (for example `3600`). 
This version assumes millimetres, which is common on metric plans.


## 2. V1 pipeline


Input image
    ↓
File validation
    ↓
Image preprocessing
    ├── grayscale
    ├── 2x upscaling
    └── thresholding
    ↓
OCR
    ├── general text
    └── dimension-focused text
    ↓
Deduplication
    ↓
Classification
    ├── room label
    ├── dimension
    ├── dimension pair
    └── scale
    ↓
Spatial association
    ↓
Application confidence
    ↓
Review flags
    ↓
JSON
    ↓
Optional ground-truth validation


## 3. Why Python?

Python was selected because the prototype requires:

- image processing
- OCR
- geometry
- regex / parsing
- JSON generation
- testing
- future PDF / AI integration

The Python ecosystem provides mature libraries for all of these tasks and allows rapid iteration.


## 4. Why EasyOCR in V1?

EasyOCR is suitable for a local prototype because it:

- is simple to integrate
- returns text, bounding boxes and confidence
- supports character allowlists
- supports rotated text handling
- can run locally

It is not assumed to be the final production OCR system.

The prototype explicitly separates the OCR component so that it can later be replaced or supplemented by:

- Azure Document Intelligence
- Google Cloud Vision / Document AI
- AWS Textract
- specialized OCR
- multimodal vision models

without redesigning the entire pipeline.

## 5. Why OpenCV?

Architectural images benefit from preprocessing because:

- dimensions may be small
- scans may be noisy
- text may intersect dimension lines
- contrast varies
- line density is high

V1 uses simple grayscale, upscaling and thresholding.

A later version can add:

- deskewing
- adaptive thresholding
- morphological cleanup
- dimension-line detection
- Hough line detection
- contour analysis
- wall / opening geometry extraction



## 6. Why deterministic parsing after OCR?

The pipeline does not trust arbitrary OCR text as a measurement.
It recognizes patterns such as:

- `3600`
- `3600 x 3300`
- `3.6 m`
- `1:100`
- `12'-6"`

This reduces the chance that drawing numbers, sheet references and identifiers are incorrectly treated as dimensions.

Further production logic can use:

- label context
- dimension-line topology
- drawing zone
- drawing type
- expected value ranges
- unit systems

## 7. Confidence model

OCR confidence alone is not sufficient.

V1 combines:

- OCR confidence
- association confidence

Overall confidence is `0.6 * OCR confidence + 0.4 * association confidence`.
For overall dimensions, association confidence is fixed at `1.0`.
The purpose is to demonstrate that low-confidence values should be retained with warnings rather than silently discarded.

## 8. Output schema justification

The JSON preserves:

### Identity
- plan ID
- measurement ID
- associated element ID

### Measurement
- type
- value
- secondary value
- unit
- normalized value

### Evidence
- raw OCR text
- source filename
- page number
- bounding box
- extraction pass

### Interpretation
- associated element
- orientation
- source type

### Reliability
- OCR confidence
- association confidence
- overall confidence
- ambiguity
- review_required
- warnings

### Plan context
- scale
- units
- detected spaces
- extraction summary
- validation metrics

## 9. AAlternatives compared

### A. OCR only

Advantages:
- fast
- inexpensive
- simple

Weaknesses:
- does not understand which object a measurement belongs to
- poor handling of dimension lines
- may miss rotated text
- may confuse IDs with measurements

Decision:
- used only as one layer, not the complete solution.

### B. Traditional CV only

Advantages:
- deterministic
- good for lines / geometry
- explainable

Weaknesses:
- weak semantic understanding
- hard to generalize across plan styles

Decision:
- used for preprocessing now; geometry extraction later.

### C. Multimodal LLM only

Advantages:
- strong semantic interpretation
- fast to prototype

Weaknesses:
- difficult to guarantee completeness
- may infer or hallucinate
- can be costly
- harder to independently validate every measurement

Decision:
- not selected as the primary extraction engine for V1. A later version may use a vision model as a secondary validator or semantic-association layer.

### D. Document AI services

Advantages:
- stronger OCR / layout handling
- managed service
- potentially better scan performance

Weaknesses:
- API cost
- cloud dependency
- possible data-governance requirements

Decision:
- strong production candidate, but EasyOCR is used first to keep the prototype local and transparent.

### E. Native PDF parsing

Advantages:
- preserves text coordinates and vector information
- avoids unnecessary OCR on digital PDFs

Weaknesses:
- not applicable to pure scans
- PDF structure varies widely

Decision:
- high-priority V2/V3 adapter.

### F. CAD parsing

Advantages:
- can access native entities, layers and dimensions
- potentially much higher geometric accuracy

Weaknesses:
- format complexity
- DWG support/licensing/tooling concerns
- different layer conventions

Decision:
- architecture will add CAD-specific adapters rather than rasterizing every CAD drawing.


## 10. Future input architecture
                    Input
                      │
         ┌────────────┼─────────────┐
         │            │             │
       Image        PDF           CAD
         │            │             │
         ▼            ▼             ▼
   ImageAdapter   PDFAdapter    CADAdapter
                      │             │
          ┌───────────┴──────┐      │
          │                  │      │
     Native PDF         Scanned PDF │
       vectors              OCR     │
          │                  │      │
          └──────────┬───────┘      │
                     │              │
                     └──────┬───────┘
                            ▼
                  Common Measurement Model


## 11. Production improvements

## Future improvements

- Support more input types: PDFs, scanned PDFs, DWG/DXF CAD files.
- Extract architectural elements: walls and wall thickness, doors, windows, openings, stairs, lifts, and their locations/orientations.
- Use geometry detection to connect dimensions to the correct walls, rooms, and openings.
- Improve accuracy through cross-checking, duplicate/conflict detection, and stronger validation before output.
- Reduce human review by automatically resolving high-confidence cases and sending only uncertain cases for review.
- Build and evaluate against a larger benchmark dataset to measure progress toward high accuracy.

### Ground truth converter — schema flexibility

Currently the converter assumes a fixed CSV schema (Space/Length/
Width). A more robust version would:
- Match column headers against a known-alias list (e.g. "Room",
  "Room Name", "Length (mm)") rather than requiring exact text,
  to handle minor naming variation without code changes.
- Fall back to column-type detection (text column = name, numeric
  columns = dimensions) when headers don't match any known alias,
  which was tested and works reliably for identifying which
  columns hold dimensions.
- For the genuinely unsolvable case — distinguishing length from
  width among multiple numeric columns with no header hint — fail
  loudly with a clear message asking for manual column mapping,
  rather than guessing and risking silently swapped values.
  
### Performance

Current per-image processing time is approximately 2-3 minutes on CPU.
This is driven by three factors:

1. EasyOCR's model weights are loaded fresh on every CLI invocation
   (no persistent process/service to keep the model warm).
2. Two OCR passes run per image (general text + dimension-focused),
   doubling inference time versus a single pass.
3. GPU acceleration is disabled by default (`gpu: false` in config.yaml);
   all inference currently runs on CPU.

For a prototype processing one plan at a time, this is acceptable.
For a production system feeding a downstream material-calculation
pipeline, this would need addressing before scaling to batches of
plans, via:
- keeping the OCR model loaded in a persistent service instead of
  reloading it per invocation
- enabling GPU inference where available
- evaluating whether both OCR passes are needed for every plan, or
  only triggered conditionally

### Approach considered: machine learning / trained room-detection model

Having previously worked with machine learning, I considered whether
a trained model could replace the rule-based OCR-plus-proximity
approach used in this pipeline. In brief, machine learning works by
showing a model many labeled examples until it learns to recognize
the underlying pattern on its own — the same way a model can learn
to recognize handwritten digits after seeing thousands of examples.
Applied here, a model could in principle be trained on many floor
plans to directly learn "this region is a room" and "this number
belongs to that room," rather than relying on hand-written distance
and alignment rules.

This isn't hypothetical — pretrained models built exactly for this
already exist, most notably CubiCasa5K, trained on ~5,000 annotated
floor plans to detect walls and room boundaries directly from images.
I investigated using this as a drop-in alternative to the current
approach.

I chose not to adopt it for this prototype, for three concrete
reasons:

1. **Training data**: building a reliable model from scratch would
   require hundreds to thousands of accurately labeled floor plans —
   the same manual verification work I did to build ground truth for
   5 plans in this project, just at a much larger scale.
2. **Compute**: training (and even running inference with) these
   models requires significantly more memory and typically GPU
   hardware, beyond what my current development machine supports —
   directly evidenced by the memory constraints already documented
   in this project's Performance/Hardware Limitations section.
3. **Maturity**: even using the pretrained CubiCasa5K model directly
   rather than training a new one, published implementations report
   inconsistent room-detection accuracy across different plan
   styles — this remains an active research problem, not a reliable
   drop-in solution today.

Given these constraints, the rule-based approach in this prototype
was the more tractable choice for a working demonstration. A trained
model remains a strong longer-term direction — see "Future
Improvements" for how the current confidence-scoring and
review-flagging system could double as a source of verified training
data over time.



## Evaluation and validation

The system is evaluated using a manually verified ground-truth file for each input plan. The ground truth lists the expected room measurements in millimetres.

After extraction, each detected measurement is compared with the ground truth using:

- room/element name;
- primary and secondary dimension values;
- a numeric tolerance of 5 mm.

For example, if ground truth expects:

BEDROOM 1: 3600 x 3300 mm
and the application extracts:
BEDROOM 1: 3600 x 3300 mm
it is counted as a matched measurement.

The evaluation reports
- `expected`: number of measurements in ground truth;
- `matched`: correctly extracted measurements;
- `missed`: expected measurements not found;
- `unexpected`: extra extracted measurements with no ground-truth match;
- `incorrect_value`: measurement found for the correct room but with the wrong value;
- `incorrect_unit`: a likely unit-conversion error;
- `duplicate_count`: possible duplicate measurements found by the application.

### Metrics

- **Precision**: of all extracted measurements, how many were correct.
  precision = matched / extracted measurements

- **Recall**: of all expected measurements, how many were found.
  recall = matched / expected measurements

- **F1 score**: one balanced score combining precision and recall.
  F1 = 2 × precision × recall / (precision + recall)

- **Accuracy**: the proportion of correct results after counting missed and unexpected measurements as errors.
### Example
If the ground truth contains 7 measurements and the application extracts 8 measurements:

Matched:    5
Missed:     2
Unexpected: 3

Then:

Precision = 5 / 8  = 0.625  (62.5%)
Recall    = 5 / 7  = 0.7143 (71.43%)
F1        = 0.6667 (66.67%)
Accuracy  = 5 / (5 + 2 + 3) = 0.5 (50%)

This evaluation ensures that the application output is measured for reliability, rather than assuming every extracted value is correct.

## Issues found and work in progress

After completing this version I reviewed the code and found several issues. I am testing fixes in a separate copy of the project and measuring each one against the ground truth before moving it here. None of the following is part of this submitted version.

1. **Room-name list.** Only room names in a fixed list are recognised, so an unfamiliar room type such as a porch would be dropped silently. In the working copy, replacing the list with a check on the text's shape gave identical scores on all five plans, so this is a robustness improvement and not an accuracy gain. The same testing showed that a confidence cut-off for room labels dropped `GUEST ROOM` (confidence 0.737) on Plan 5, so that cut-off needs tuning.
2. **Number of OCR runs.** This version reads the full plan three times and also runs a dimension pass, an edge pass and two small crops, up to seven OCR calls per plan. On Plan 2 the three full reads returned the same detections as a single read, and using one read took the run from about 81 seconds to about 49 seconds with identical results.
3. **Rotation setting.** With rotation enabled, some horizontal text lines (for example `UTILITY` and `3600 x 3300` on Plan 1) are misread as `1`. With rotation off they are read correctly, but the vertical dimension numbers along the walls are misread and Plan 4 gets worse. Wide boxes and tall boxes behave differently, so I am testing choosing the reading according to the box shape.
4. **Missing-dimension warning.** A room that ends up with no dimensions gets no warning in this version (for example `DINING` on Plan 5). The working copy adds one.


## AI-Assisted Development

GitHub Copilot was used directly in the editor for generating boilerplate,
implementing functions from a description, and proposing fixes to
specific bugs. For debugging, understanding unfamiliar parts of the
generated code line-by-line, and researching alternative technical
approaches, I also used conversational AI assistants (Claude and
ChatGPT) as a sounding board — this was research and verification, not
code generation; Copilot remained the tool actually writing code in
this repository.

I did not accept AI-generated code without verifying it. Two concrete
examples where an initial AI-suggested fix was wrong and had to be
corrected after testing: