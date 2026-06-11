## Project Development Rule

These rules are MANDATORY. Before declaring ANY task done, re-read this file and verify
you have respected every rule below. Do not skip a rule because it "looks minor".

When working with projects:

1. **Project Context Guide**: Always check the links/sources provided for the current task
   to access the relevant source code repositories before implementing.

   - The code should be written in Python and, to provide a UI, use the Streamlit framework.

2. **Project Structure**: Follow these conventions:

   - The documents of the project should be created in the "Docs" folder except README.md.

   - Always provide a Mermaid flow architecture for the project in the "Architecture.md" file.

   - All the BASH scripts, if needed, should be written in the "scripts" folder.

   - All the input documents are to be found in the "input" folder.

   - All the output documents which are asked to be provided should be written in timestamped
     format in the "output" folder.

   - The result documents should be written in the "output" folder; if the "output" folder
     does not exist, it should be created.

   - Always provide README.md with architecture + workflow diagrams as described.

   - Always provide a ".gitignore" file which filters/ignores any ".env" files or any folders
     whose names start with "_" (underscore) (e.g.: _sources/, _images/, _docs/...).

3. **Documentation discipline** (apply strictly — this is where mistakes happen):

   - When you make updates / enhancements and/or fix bugs, UPDATE the existing documents and
     scripts. Do NOT create new ones.

   - Never create a second document covering a topic an existing document already covers.
     Extend the existing one.

   - Never leave a document describing an outdated state of the project. If the project
     evolves, update EVERY affected doc (README, Architecture.md, plan/checklist, etc.) so
     they all stay consistent with the current code. No stale or contradicting docs.

4. **Key Patterns**:

   - For Python applications, always work in a single virtual environment, and make sure it
     contains everything required to run the full project.

   - Always test the functionality of the code you provide: actually run it end-to-end and
     confirm it works before declaring the task done.

5. **Misc**:

   - Detect the OS you're running on to provide ad-hoc scripts if needed.

   - On a macOS platform, don't use port 5000 (reserved for the "AirDrop" application).

6. **Definition of Done** — before saying a task is finished, confirm:

   - The code was actually run and works end-to-end.
   - No new/duplicate document was created where an existing one should have been updated.
   - No document is left outdated; all docs are consistent with the current code.
   - All rules above are respected.