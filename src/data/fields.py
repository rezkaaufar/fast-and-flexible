import logging

import torchtext


class MechField(torchtext.data.Field):
    """Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True.
    """

    def __init__(self, **kwargs):
        """Initialize the datafield, but force batch_first and include_lengths to be True.
        Args:
            **kwargs: Description
        """
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning(
                "Option batch_first has to be set to use machine.  Changed to True.")
        kwargs['batch_first'] = True

        if kwargs.get('include_lengths') is False:
            logger.warning(
                "Option include_lengths has to be set to use machine.  Changed to True.")
        kwargs['include_lengths'] = True

        super(MechField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(MechField, self).build_vocab(*args, **kwargs)