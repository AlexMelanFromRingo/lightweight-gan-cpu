#!/usr/bin/env python3
"""
Test script to verify checkpoint save/load compatibility with torch.compile
This specifically tests the fix for the _orig_mod prefix issue.
"""
import torch
import os
import shutil
from lightweight_gan import LightweightGAN

def test_checkpoint_save_load_with_compile():
    """Test that checkpoints can be saved and loaded with torch.compile enabled"""
    print("=" * 60)
    print("Testing checkpoint compatibility with torch.compile")
    print("=" * 60)

    # Create temporary directories
    test_models_dir = './test_models_temp'
    test_results_dir = './test_results_temp'

    # Clean up if exists
    if os.path.exists(test_models_dir):
        shutil.rmtree(test_models_dir)
    if os.path.exists(test_results_dir):
        shutil.rmtree(test_results_dir)

    os.makedirs(test_models_dir, exist_ok=True)
    os.makedirs(test_results_dir, exist_ok=True)

    try:
        print("\n[1/5] Creating GAN with torch.compile enabled...")
        gan1 = LightweightGAN(
            latent_dim=128,
            image_size=64,
            optimizer='adam',
            fmap_max=256,
            lr=2e-4,
            rank=0,
            use_compile=True  # Enable torch.compile
        )
        print("✓ GAN created with torch.compile")

        # Check if models are compiled
        is_compiled = hasattr(gan1.G, '_orig_mod')
        print(f"✓ Models are compiled: {is_compiled}")

        print("\n[2/5] Running forward pass to initialize models...")
        device = next(gan1.G.parameters()).device
        latents = torch.randn(2, 128, device=device)

        with torch.no_grad():
            output1 = gan1.G(latents)
        print(f"✓ Generator output shape: {output1.shape}")

        # Save initial weights for comparison
        if is_compiled:
            initial_weight = gan1.G._orig_mod.initial_conv[0].weight.data.clone()
        else:
            initial_weight = gan1.G.initial_conv[0].weight.data.clone()

        print("\n[3/5] Saving checkpoint...")
        checkpoint_path = os.path.join(test_models_dir, 'test', 'model_1.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Manually save using the same logic as Trainer
        def get_state_dict(model):
            if hasattr(model, '_orig_mod'):
                return model._orig_mod.state_dict()
            return model.state_dict()

        gan_state = {}
        for key in ['G', 'D', 'GE', 'G_opt', 'D_opt']:
            if hasattr(gan1, key):
                attr = getattr(gan1, key)
                if hasattr(attr, 'state_dict'):
                    if key in ['G', 'D', 'GE']:
                        gan_state[key] = get_state_dict(attr)
                    else:
                        gan_state[key] = attr.state_dict()

        save_data = {
            'GAN': gan_state,
            'version': '1.2.1'
        }

        torch.save(save_data, checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}")

        # Check saved keys
        loaded_check = torch.load(checkpoint_path, weights_only=True)
        sample_keys = list(loaded_check['GAN']['G'].keys())[:3]
        print(f"✓ Sample keys in checkpoint: {sample_keys}")
        has_orig_mod = any('_orig_mod' in k for k in loaded_check['GAN']['G'].keys())
        print(f"✓ Checkpoint has _orig_mod prefix: {has_orig_mod}")

        print("\n[4/5] Creating new GAN instance with torch.compile...")
        gan2 = LightweightGAN(
            latent_dim=128,
            image_size=64,
            optimizer='adam',
            fmap_max=256,
            lr=2e-4,
            rank=0,
            use_compile=True  # Enable torch.compile
        )
        print("✓ New GAN instance created")

        print("\n[5/5] Loading checkpoint into new instance...")
        # Load the checkpoint using new format
        load_data = torch.load(checkpoint_path, weights_only=True)
        gan_state = load_data['GAN']

        # New format - load each component separately
        for key in ['G', 'D', 'GE']:
            if key in gan_state and hasattr(gan2, key):
                model = getattr(gan2, key)
                # Handle compiled models
                if hasattr(model, '_orig_mod'):
                    model._orig_mod.load_state_dict(gan_state[key], strict=True)
                else:
                    model.load_state_dict(gan_state[key], strict=True)

        for key in ['G_opt', 'D_opt']:
            if key in gan_state and hasattr(gan2, key):
                getattr(gan2, key).load_state_dict(gan_state[key])

        print("✓ Checkpoint loaded successfully")

        print("\n[6/6] Verifying weights match...")
        if hasattr(gan2.G, '_orig_mod'):
            loaded_weight = gan2.G._orig_mod.initial_conv[0].weight.data
        else:
            loaded_weight = gan2.G.initial_conv[0].weight.data

        weight_match = torch.allclose(initial_weight, loaded_weight, rtol=1e-5)
        print(f"✓ Weights match: {weight_match}")

        if not weight_match:
            print("✗ ERROR: Weights don't match after loading!")
            return False

        print("\n" + "=" * 60)
        print("✓ All tests PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("\nCleaning up test directories...")
        if os.path.exists(test_models_dir):
            shutil.rmtree(test_models_dir)
        if os.path.exists(test_results_dir):
            shutil.rmtree(test_results_dir)
        print("✓ Cleanup complete")


def test_old_checkpoint_format():
    """Test loading old format checkpoints (with _orig_mod prefix in keys)"""
    print("\n" + "=" * 60)
    print("Testing OLD checkpoint format compatibility")
    print("=" * 60)

    test_models_dir = './test_models_temp'
    test_results_dir = './test_results_temp'

    # Clean up if exists
    for dir_path in [test_models_dir, test_results_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    try:
        print("\n[1/4] Creating GAN with torch.compile...")
        gan1 = LightweightGAN(
            latent_dim=128,
            image_size=64,
            optimizer='adam',
            fmap_max=256,
            lr=2e-4,
            rank=0,
            use_compile=True
        )

        # Initialize with forward pass
        device = next(gan1.G.parameters()).device
        latents = torch.randn(2, 128, device=device)
        with torch.no_grad():
            _ = gan1.G(latents)

        # Save initial weights
        if hasattr(gan1.G, '_orig_mod'):
            initial_weight = gan1.G._orig_mod.initial_conv[0].weight.data.clone()
        else:
            initial_weight = gan1.G.initial_conv[0].weight.data.clone()

        print("\n[2/4] Creating OLD format checkpoint...")
        checkpoint_path = os.path.join(test_models_dir, 'test', 'model_1.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # OLD save format - full LightweightGAN state_dict with _orig_mod prefixes
        old_save_data = {
            'GAN': gan1.state_dict(),  # Full state dict with _orig_mod prefixes
            'version': '1.2.0'
        }

        torch.save(old_save_data, checkpoint_path)
        print("✓ Old format checkpoint created")

        # Verify it has _orig_mod prefix
        loaded = torch.load(checkpoint_path, weights_only=True)
        sample_keys = list(loaded['GAN'].keys())[:3]
        print(f"  Sample keys: {sample_keys}")
        has_prefix = any('_orig_mod' in k for k in loaded['GAN'].keys())
        print(f"✓ Has _orig_mod prefix: {has_prefix}")

        print("\n[3/4] Creating new GAN and testing load logic...")
        gan2 = LightweightGAN(
            latent_dim=128,
            image_size=64,
            optimizer='adam',
            fmap_max=256,
            lr=2e-4,
            rank=0,
            use_compile=True
        )

        # Test the load logic from the Trainer's load method
        load_data = torch.load(checkpoint_path, weights_only=True)
        gan_state = load_data['GAN']

        # This is the OLD format handling from Trainer.load()
        state_dict = gan_state

        # Helper function to extract state dict for a specific component
        def extract_component_state(full_state, prefix):
            component_state = {}
            prefix_with_dot = prefix + '.'
            for k, v in full_state.items():
                if k.startswith(prefix_with_dot):
                    # Remove component prefix and _orig_mod if present
                    key = k[len(prefix_with_dot):]  # Remove "G." or "D." etc
                    key = key.replace('_orig_mod.', '')  # Remove _orig_mod prefix
                    component_state[key] = v
            return component_state

        # Load each model component
        for key in ['G', 'D', 'GE']:
            if hasattr(gan2, key):
                model = getattr(gan2, key)
                component_state = extract_component_state(state_dict, key)

                if len(component_state) > 0:
                    # Load into _orig_mod if model is compiled, otherwise load normally
                    if hasattr(model, '_orig_mod'):
                        model._orig_mod.load_state_dict(component_state, strict=True)
                    else:
                        model.load_state_dict(component_state, strict=True)

        print("✓ Old format checkpoint loaded successfully")

        print("\n[4/4] Verifying weights match...")
        if hasattr(gan2.G, '_orig_mod'):
            loaded_weight = gan2.G._orig_mod.initial_conv[0].weight.data
        else:
            loaded_weight = gan2.G.initial_conv[0].weight.data

        weight_match = torch.allclose(initial_weight, loaded_weight, rtol=1e-5)
        print(f"✓ Weights match: {weight_match}")

        if not weight_match:
            print("✗ ERROR: Weights don't match after loading!")
            return False

        print("\n" + "=" * 60)
        print("✓ Old format compatibility test PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        for dir_path in [test_models_dir, test_results_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)


if __name__ == "__main__":
    import sys

    print("Testing checkpoint save/load compatibility with torch.compile")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    # Test 1: New format
    test1_passed = test_checkpoint_save_load_with_compile()

    # Test 2: Old format compatibility
    test2_passed = test_old_checkpoint_format()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"New checkpoint format: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Old checkpoint format: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print("=" * 60)

    if test1_passed and test2_passed:
        print("\n🎉 All checkpoint compatibility tests PASSED!")
        sys.exit(0)
    else:
        print("\n⚠ Some tests FAILED")
        sys.exit(1)
