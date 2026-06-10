## Project Development Rule

When working with projects:

1. **Project Context Guide**: Always check the links provided here to access the source code repositories for implementation:

   - Getting Started with Docling: https://github.com/docling-project/docling/blob/main/README.md

   - The Docling Project  GitHub repository: https://github.com/docling-project/docling

   - The code should be written in Python and to provide a UI you should use the Streamlit framework

2. **Project Structure**: Follow these conventions:

   - The documents of the project should be created in "Docs" folder except readme.md

   - Always provide a Mermaid flow architecture for the project in the "Architecture.md" file

   - All the BASH scripts if needed, should be written in "scripts" folder

   - All the input documents are to be found in "input" folder

   - All the output documents which are asked to be provided should be writen in timestamped format in "output" folder

   - The result documents should be written in "output" folder, if the "output" folder does not exist, it should be created

   - Always provide README.md with architecture + workflow diagrams as described

   - Always provide a ".gitignore" file which filters/ignores any ".env" files or any folders whichs' names start with "\_" (underscore) to be pushed to GitHub (e.g.: \_sources/, \_images/, \_docs/... )

3. **Key Patterns**:

   - For Python applications always work in a virtual environment

   - Always test the functionnality of the code you provide

   - When you make updates/enhancements and/or correct the bugs, update the existing documents and scripts, don't create new ones

4. **Misc**:

   - Detect the OS you’re running on to provide ad-hoc scripts if needed

   - On a MacOS platform, don't use the port 5000, it is reserved for the "AirDrop" application