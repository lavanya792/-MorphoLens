MorphoLens dashboard fixes
/Users/lavanya/Desktop/MorphoLens_Dashboard_Fixes/INSTALL_DELETE_IMAGES.md
Copy/replace:
- frontend/src/App.tsx
- frontend/src/style.css
- backend/app/routes/dashboard.py

The dashboard now requests the original image from:
GET /api/dashboard/analysis/{analysis_id}/image

It requests mask and overlay by analysis ID. The route resolves files by matching the upload UUID in the stored image_path against files in backend/uploads and backend/outputs. If your upload service names outputs without the same UUID prefix, adjust _analysis_output() to your actual output filenames.

Delete:
DELETE /api/dashboard/analysis/{analysis_id}
This deletes the analysis row, its nuclei_features and analysis_metrics rows, plus files whose names start with the image-path stem in backend/uploads and backend/outputs. Confirm the naming convention before using delete on important data. Back up backend/data/morpholens.db first.

Restart backend and Vite after replacing files.
