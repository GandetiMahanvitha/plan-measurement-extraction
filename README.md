# Building Plan Measurement Extraction Prototype

A local Python prototype for the first stage of a building-plan measurement system.

## Goal

The prototype accepts an architectural image as input and produces a structured JSON result file named like:

`<input_name>_result.json`

The output is designed to preserve the measurement context needed by a downstream  workflow, including:

- what was measured
- value and unit
- associated building element / room
- source location within the plan
- OCR and association confidence
- review and ambiguity flags
- raw OCR evidence
- detected drawing scale
- extraction warnings and summary statistics

This version focuses on image-based floor plans with explicit dimension annotations and is intentionally a research-style prototype rather than a production-grade extraction engine.

---

## Recommended environment

- Python 3.11+
- Windows, macOS, or Linux
- CPU is sufficient for the current prototype
- GPU may be enabled later for OCR acceleration
> EasyOCR downloads model weights on first use if they are not already cached.

---

## Project structure

building_plan_measurement_extractor/
├── .gitignore
├── check_current_scores.py
├── config.low-memory.yaml
├── config.yaml
├── data/
│   ├── ground_truth/
│   │   ├── plan1_ground_truth.json
│   │   ├── plan1_table.csv
│   │   ├── plan2_ground_truth.json
│   │   ├── plan2_table.csv
│   │   ├── plan3_ground_truth.json
│   │   ├── plan3_table.csv
│   │   ├── plan4_ground_truth.json
│   │   ├── plan4_table.csv
│   │   ├── plan5_ground_truth.json
│   │   └── plan5_table.csv
│   ├── input/
│   │   ├── plan1.jpeg
│   │   ├── plan2.jpeg
│   │   ├── plan3.jpeg
│   │   ├── plan4.jpeg
│   │   └── plan5.png
│   └── output/
│       ├── plan1_result.json
│       ├── plan2_result.json
│       ├── plan3_result.json
│       ├── plan4_result.json
│       └── plan5_result.json
├── docs/
│   ├── TECHNICAL_APPROACH.md
├── pyproject.toml
├── README.md
├── requirements.txt
├── run.py
├── schemas/
│   └── result.schema.json
├── scripts/
│   ├── csv_to_ground_truth.py
│   └── run_batch.py
├── src/
│   ├── building_plan_measurement_extractor.egg-info/
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   ├── dependency_links.txt
│   │   ├── requires.txt
│   │   └── top_level.txt
│   └── plan_extractor/
│       ├── __init__.py
│       ├── association.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── ocr_engine.py
│       ├── parsing.py
│       ├── pipeline.py
│       ├── preprocessing.py
│       └── validation.py
└── tests/
    ├── test_deduplication_with_offset.py
    ├── test_parsing.py
    ├── test_room_dimension_merge.py
    └── test_validation.py




# 1. Installation
Create a virtual environment.

### Windows
python -m venv .venv
.venv\Scripts\activate


### macOS / Linux
python -m venv .venv
source .venv/bin/activate
 

Install packages:
python -m pip install --upgrade pip
pip install -r requirements.txt
 

# 2. Add an input plan

Place an image in: data/input/
Supported by the current CLI:
- PNG
- JPG
- JPEG
- TIFF
- BMP

# 3. Run the extractor
python run.py data/input/plan1.jpeg
Optional arguments supported by the current CLI:
python run.py data/input/plan1.jpeg --output-dir data/output
python run.py data/input/plan1.jpeg --config config.low-memory.yaml
python run.py data/input/plan1.jpeg --ground-truth data/ground_truth/plan1_ground_truth.json
The application writes the result as: data/output/plan1_result.json
 
The current argument set is:

- positional `input` (required)
- `--output-dir` (default: `data/output`)
- `--config` (optional YAML config file)
- `--ground-truth` (optional JSON evaluation file)


# 4. Run with a ground-truth file

python run.py data/input/plan1.jpeg --ground-truth data/ground_truth/plan1_ground_truth.json
If a ground-truth file is supplied, the result includes evaluation data such as:

- expected count
- matched count
- missed count
- unexpected count
- precision
- recall
- F1 score


# 5. Run tests
pytest -q
 

# 6. Processing pipeline

Image
  ↓
File validation
  ↓
OpenCV preprocessing
  ├── original
  ├── grayscale/upscaled
  └── thresholded
  ↓
EasyOCR
  ├── general-text pass
  └── dimension-focused pass
  ↓
Deduplication
  ↓
Text classification
  ├── room / space labels
  ├── explicit dimension values
  ├── dimension pairs
  ├── scale
  └── other text
  ↓
Spatial association
  ↓
Application confidence
  ↓
Warnings / review flags
  ↓
Structured JSON
  ↓
Optional ground-truth evaluation
 


# 7. Why two OCR passes?

Architectural plans contain different kinds of text.

A general OCR pass is useful for:

- BEDROOM 1
- KITCHEN
- LIVING / DINING
- W1 / D1
- notes

A measurement-focused OCR pass restricts recognition to common measurement characters and improves the chance of reading values such as:

- `3600`
- `2400 x 2100`
- `1:100`
- `12'-6"`
- `3.6 m`

The pipeline merges and deduplicates the outputs from those passes.


# 8. Output 

A downstream material-calculation system should not receive only a minimal string like:

 json :
{
  "bedroom": "3600x3300"
}

That drops too much evidence.
Instead, each measurement preserves:

- normalized value
- raw OCR text
- source type
- location
- orientation
- associated element
- OCR confidence
- association confidence
- overall confidence
- review status
sample one is in schemas/result.schema.json

## Dependencies

The project currently declares the following dependencies in `requirements.txt`
pip install -r requirements.txt
pip install -e .

# 9. Reliability approach
The prototype avoids treating OCR confidence as the only truth signal.

Example:
OCR confidence:             0.68
Spatial association:        0.92
Application confidence:     0.78

The overall confidence is calculated as `0.6 * OCR confidence + 0.4 * association confidence`.
 
A lower-confidence but syntactically valid measurement is retained and marked for review instead of silently discarded.


# 10. Known V1 limitations

The current version is intentionally limited.

It does not yet reliably:

- reconstruct walls as architectural objects
- understand arbitrary dimension-line topology
- calculate missing lengths solely from drawing scale
- parse multi-view sheets
- parse elevations and sections fully
- parse DWG/DXF
- interpret handwritten annotations
- resolve conflicting dimensions
- understand every imperial notation variant
- guarantee that every measurement in a complex plan is found
- process a single plan in under a minute (current per-image time is
  approximately 2-3 minutes on CPU; see TECHNICAL_APPROACH.md for why
  and how this could be optimized)

These are documented in more detail in `docs/TECHNICAL_APPROACH.md`.


# 11. Planned evolution

## V1 — current
Images + explicit text dimensions + OCR evidence + structured JSON.

## V2
Dimension-line detection, stronger room association, geometry checks, annotated output.

## V3
Native PDF extraction using vector/text information before raster OCR.

PyMuPDF can be introduced as a PDF adapter so that native PDF words and their coordinates can be used directly, while scanned pages can be rendered and passed through the image pipeline.

## V4
Scale-calibrated calculated dimensions, wall thickness, doors/windows/openings, elevations.

## V5
DWG/DXF adapters, CAD entity extraction, multi-view reconciliation, stronger validation and human-review tooling.


# 12. Ground Truth & Validation

To validate extraction accuracy against a known-correct answer key:

1. Manually read the plan's dimensions and type them into a CSV:
   `data/ground_truth/<plan_name>_table.csv` (columns: Space, Length, Width)
2. Convert it to the JSON format the validator expects:
   `python scripts/csv_to_ground_truth.py data/ground_truth/<plan_name>_table.csv data/ground_truth/<plan_name>_ground_truth.json <PLAN_ID>`
3. Run extraction with validation enabled:
   `python run.py data/input/<plan_name>.png --ground-truth data/ground_truth/<plan_name>_ground_truth.json`
