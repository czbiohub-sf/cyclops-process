import tifffile
import json
import pprint # For pretty printing complex dictionaries

# --- IMPORTANT: Change this to your file path ---
file_path = '/path/to/your/ndtiff_dataset.tif'
# ---------------------------------------------

print(f"Inspecting file: {file_path}\n")

try:
    with tifffile.TiffFile(file_path) as tif:
        # 1. Check if Micro-Manager metadata exists. This is the most common place.
        if tif.micromanager_metadata:
            print("--- Found Micro-Manager Metadata ---")
            
            # The metadata is a dictionary. Let's look at its main keys first.
            mm_meta = tif.micromanager_metadata
            print(f"Top-level keys: {list(mm_meta.keys())}")

            # 2. The channel names are almost always in the 'Summary' key.
            #    This 'Summary' is usually a JSON STRING, so we need to parse it.
            if 'Summary' in mm_meta:
                print("\nAttempting to parse 'Summary' metadata...")
                summary_data = json.loads(mm_meta['Summary'])
                
                # 3. Now we have a proper dictionary. Look for channel names.
                #    Common keys are 'ChNames', 'ChannelNames', or 'channel-names'.
                
                if 'ChNames' in summary_data:
                    print(f"\nSUCCESS: Found channel names in 'ChNames'!")
                    print("Channels:", summary_data['ChNames'])
                
                elif 'ChannelNames' in summary_data:
                    print(f"\nSUCCESS: Found channel names in 'ChannelNames'!")
                    print("Channels:", summary_data['ChannelNames'])
                    
                else:
                    print("\nCould not find 'ChNames' or 'ChannelNames' directly.")
                    print("Dumping all summary keys for you to inspect:")
                    pprint.pprint(summary_data)
            else:
                print("\nNo 'Summary' key found in Micro-Manager metadata.")
                print("Dumping all Micro-Manager metadata for inspection:")
                pprint.pprint(mm_meta)

        else:
            print("--- No Micro-Manager metadata found. ---")
            # As a fallback, you can try to inspect ImageJ metadata if it exists
            if tif.imagej_metadata:
                print("\nFound ImageJ Metadata. Dumping for inspection:")
                pprint.pprint(tif.imagej_metadata)


except FileNotFoundError:
    print(f"ERROR: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")