from tensorflow.test import TestCase
from groco.groups import wallpaper_group_dict
import tensorflow as tf


class TestWallpaperGroup(TestCase):
    def test_inverse_comp(self):
        """
        The composition attribute gives the composition of an inverse with another group element,
        so for the inverse to be correct the diagonal needs to be the identity.
        """
        for group in wallpaper_group_dict.values():
            identities = tf.reshape(
                tf.concat([group.composition[r][c] for r, c in enumerate(group.inverses)], axis=0),
                (group.order,))
            self.assertAllEqual(identities, tf.zeros(group.order, dtype=tf.int32))

    def test_subgroups(self):
        """
        A subset H of a group G is a subgroup if and only if for all g, h in H, g^-1 h is also in H.
        """
        for group in wallpaper_group_dict.values():
            for subgroup_name, subgroup_indices in group.subgroup.items():
                subgroup_invs = [group.inverses[i] for i in subgroup_indices]
                subgroup_composition = tf.gather(group.composition, axis=0, indices=subgroup_invs)
                subgroup_composition = tf.gather(subgroup_composition, axis=1, indices=subgroup_indices)
                elements, _ = tf.unique(tf.reshape(subgroup_composition, [-1]))
                elements = tf.sort(elements)

                msg = f'Subgroup {subgroup_name} not closed in group {group.name}'
                self.assertAllEqual(elements, tf.sort(subgroup_indices), msg=msg)

    def test_cosets_identity(self):
        """Cosets contain identity."""
        for group in wallpaper_group_dict.values():
            for coset_name, coset in group.cosets.items():
                msg = f'Coset {coset_name} of group {group.name} does not contain identity.'
                self.assertEqual(tf.reduce_min(coset), tf.constant([0]), msg=msg)

    def test_cosets_size(self):
        """Number of cosets times corresponding group order equals the full group order."""
        for group in wallpaper_group_dict.values():
            for coset_name, coset in group.cosets.items():
                subgroup = wallpaper_group_dict[coset_name]

                msg = f'Cosets of subgroup {coset_name} of group {group.name} not the right size.'
                self.assertEqual(group.order, subgroup.order * len(coset), msg=msg)

    def test_cosets_unique(self):
        """
        Check that multiplying the subgroup with its cosets recovers the full group.
        """
        for group in wallpaper_group_dict.values():
            for coset_name, coset in group.cosets.items():
                subgroup_indices = group.subgroup[coset_name]
                products = tf.gather(group.composition, axis=1, indices=coset)
                subgroup_inverses = tf.gather(group.inverses, axis=0, indices=subgroup_indices)
                products = tf.gather(products, axis=0, indices=subgroup_inverses)
                products = tf.sort(tf.reshape(products, [-1]))

                msg = f'Subgroup {coset_name} multiplied with its cosets does not recover full group {group.name}.'
                self.assertAllEqual(tf.range(group.order), products, msg=msg)

    def test_action_composition(self):
        signal = tf.random.normal((28, 28, 3), seed=42)
        for group in wallpaper_group_dict.values():
            g_signal = group.action(signal, spatial_axes=[0, 1], new_group_axis=2)
            for gi in range(group.order):
                gi_signal = tf.gather(g_signal, axis=2, indices=[gi])
                gi_signal = tf.reshape(gi_signal, (28, 28, 3))

                h_gi_signal = group.action(gi_signal, spatial_axes=[0, 1], new_group_axis=2)
                h_gi = tf.reshape(tf.gather(group.composition, axis=1, indices=[gi]), (group.order))

                h_gi_at_signal = group.action(signal, spatial_axes=[0, 1], new_group_axis=2)
                h_gi_at_signal = tf.gather(h_gi_at_signal, axis=2, indices=h_gi)

                msg = f'Action of {group.name} not compatible with its composition.'
                self.assertAllEqual(h_gi_signal, h_gi_at_signal, msg=msg)

    def test_action_shape(self):
        signal = tf.random.normal((28, 28, 3), seed=42)
        for group in wallpaper_group_dict.values():
            g_signal = group.action(signal, spatial_axes=[0, 1], new_group_axis=0)
            self.assertEqual(g_signal.shape, (group.order) + signal.shape)

    def test_action_on_subgroup_shape(self):
        for group in wallpaper_group_dict.values():
            for subgroup_name, subgroup_indices in group.subgroup.items():
                signal = tf.random.normal((28, 28, len(subgroup_indices), 3))
                g_signal = group.action(signal, spatial_axes=[0, 1], new_group_axis=0, domain_group=subgroup_name,
                                        acting_group=group.name)
                self.assertEqual(g_signal.shape, (group.order, ) + signal.shape)

    def test_action_on_signal_composition(self):
        signal = tf.random.normal((28, 28, 3), seed=42)
        new_group_axis = 3
        for group in wallpaper_group_dict.values():
            g_signal = group._action_on_grid(signal, new_group_axis=new_group_axis, spatial_axes=[0, 1])
            for gi in range(group.order):
                gi_signal = tf.reshape(tf.gather(g_signal, axis=new_group_axis, indices=[gi]), signal.shape)
                h_gi_signal = group._action_on_grid(gi_signal, new_group_axis=new_group_axis, spatial_axes=[0, 1])
                h_gi = tf.reshape(tf.gather(group.composition, axis=1, indices=[gi]), (group.order))
                h_gi_at_signal = tf.gather(
                    group._action_on_grid(signal, new_group_axis=new_group_axis, spatial_axes=[0, 1]),
                    axis=new_group_axis, indices=h_gi)

                msg = f'Action of {group.name} not compatible with its composition.'
                self.assertAllEqual(h_gi_signal, h_gi_at_signal, msg=msg)

    def test_action_on_group_composition(self):
        new_group_axis = 3
        for group in wallpaper_group_dict.values():
            signal = tf.random.normal((28, 28, group.order, 3), seed=42)
            g_signal = group.action(signal, spatial_axes=[0, 1], group_axis=2, new_group_axis=new_group_axis)
            for gi in range(group.order):
                gi_signal = tf.reshape(tf.gather(g_signal, axis=new_group_axis, indices=[gi]), signal.shape)
                h_gi_signal = group.action(
                    gi_signal, spatial_axes=[0, 1], group_axis=2, new_group_axis=new_group_axis)
                h_gi = tf.reshape(tf.gather(group.composition, axis=1, indices=[gi]), (group.order))
                h_gi_at_signal = tf.gather(
                    group.action(signal, spatial_axes=[0, 1], group_axis=2, new_group_axis=new_group_axis),
                    axis=new_group_axis, indices=h_gi)

                msg = f'Action of {group.name} not compatible with its composition.'
                self.assertAllEqual(h_gi_signal, h_gi_at_signal, msg=msg)

    def test_subgroup_action_on_grid(self):
        signal = tf.random.normal((28, 28, 3))
        for group in wallpaper_group_dict.values():
            g_signal = group.action(signal, spatial_axes=[0, 1], new_group_axis=0)
            for subgroup_name, subgroup_indices in group.subgroup.items():
                subgroup = wallpaper_group_dict[subgroup_name]
                h_signal = subgroup.action(signal, spatial_axes=[0, 1], new_group_axis=0)
                g_signal_sub = tf.gather(g_signal, axis=0, indices=subgroup_indices)

                msg = f'Action of subgroup {subgroup_name} on signal on grid not the same as corresponding indices in action of full group {group.name}'
                self.assertAllEqual(h_signal, g_signal_sub, msg=msg)

    def test_subgroups_cosets(self):
        """Test only if the keys are the same."""
        for group in wallpaper_group_dict.values():
            self.assertAllEqual(set(group.subgroup.keys()), set(group.cosets.keys()))


tf.test.main()