package pl.robolab.fira.colortracking;

import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import java.io.UnsupportedEncodingException;
import java.lang.ref.WeakReference;


public class MainActivity extends Activity {

    // Message types sent from the BluetoothCommandService Handler
    public static final int MESSAGE_DEVICE_NAME = 3;
    public static final int MESSAGE_CONNECT_FAILED = 5;
    public static final int MESSAGE_CONNECT_LOST = 6;
    // Key names received from the BluetoothCommandService Handler
    public static final String DEVICE_NAME = "device_name";
    // Intent request codes
    private static final int REQUEST_CONNECT_DEVICE = 1;
    private static final int REQUEST_ENABLE_BT = 2;
    // The Handler that gets information back from the BluetoothCommandService
    private final Handler mHandler = new IncomingHandler(this);
    private Toast exitToast;

    // Local Bluetooth adapter
    private BluetoothAdapter mBluetoothAdapter = null;
    // Member object for Bluetooth Command Service
    private BluetoothCommandService mCommandService = null;

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK) {
            sendSpeedCommand(0, 0);
            if (exitToast.getView().isShown()) {
                exitToast.cancel();
                finish();
            } else {
                exitToast.show();
            }
            return true;
        } else if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            sendSpeedCommand(10, 20);
            Log.v("SpeedTest", "enabled");
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    @Override
    public boolean onKeyUp(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            sendSpeedCommand(0, 0);
            Log.v("SpeedTest", "disabled");
            return true;
        }
        return super.onKeyUp(keyCode, event);
    }


    private void sendSpeedCommand(int leftV, int rightV) {
        String toSend = "[=" + leftV + "," + rightV + "]";

        try {
            if (mCommandService != null) {
                mCommandService.write(toSend.getBytes("ASCII"));
            }
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//
//			other handlers
//
///////////////////////////////////////////////////////////////////////////////////   


    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.v("InitLog", "onCreate");

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        requestWindowFeature(Window.FEATURE_ACTION_BAR_OVERLAY);
        exitToast = Toast.makeText(this, R.string.press_again_exit, Toast.LENGTH_SHORT);
        // Get local Bluetooth adapter
        mBluetoothAdapter = BluetoothAdapter.getDefaultAdapter();

        Log.v("InitLog", "onCreate got Bluetooth adapter");

        // If the adapter is null, then Bluetooth is not supported
        if (mBluetoothAdapter == null) {
            Toast.makeText(this, R.string.no_bluetooth, Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        Log.v("InitLog", "onCreate: bluetooth adapter != null");
        setContentView(R.layout.main);

    }

    @Override
    protected void onStart() {
        super.onStart();
        getActionBar().setDisplayShowTitleEnabled(false);
        Log.v("InitLog", "onStart");

        // If BT is not on, request that it be enabled.
        // setupCommand() will then be called during onActivityResult
        if (!mBluetoothAdapter.isEnabled()) {
            Log.v("InitLog", "onStart: not enabled");

            Intent enableIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
            startActivityForResult(enableIntent, REQUEST_ENABLE_BT);
        }
        // otherwise set up the command service
        else {
            Log.v("InitLog", "onStart: enabled");

            if (mCommandService == null)
                setupCommand();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.v("InitLog", "onPause");
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.v("InitLog", "onResume");

        if (mCommandService != null) {
            if (mCommandService.getState() == BluetoothCommandService.STATE_NONE) {

                Log.v("InitLog", "onResume, state == NONE, before service.start()");
                mCommandService.start();
            }
        }
    }

    private void setupCommand() {
        // Initialize the BluetoothCommandService to perform bluetooth connections
        mCommandService = new BluetoothCommandService(mHandler);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (mCommandService != null)
            mCommandService.stop();
    }

    private void ensureDiscoverable() {
        if (mBluetoothAdapter.getScanMode() !=
                BluetoothAdapter.SCAN_MODE_CONNECTABLE_DISCOVERABLE) {
            Intent discoverableIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_DISCOVERABLE);
            discoverableIntent.putExtra(BluetoothAdapter.EXTRA_DISCOVERABLE_DURATION, 300);
            startActivity(discoverableIntent);
        }
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode) {
            case REQUEST_CONNECT_DEVICE:
                // When DeviceListActivity returns with a device to connect
                if (resultCode == Activity.RESULT_OK) {
                    // Get the device MAC address
                    String address = data.getExtras()
                            .getString(DeviceListActivity.EXTRA_DEVICE_ADDRESS);
                    // Get the BluetoothDevice object
                    BluetoothDevice device = mBluetoothAdapter.getRemoteDevice(address);
                    // Attempt to connect to the device
                    mCommandService.connect(device);
                }
                break;
            case REQUEST_ENABLE_BT:
                // When the request to enable Bluetooth returns
                if (resultCode == Activity.RESULT_OK) {
                    // Bluetooth is now enabled, so set up a chat session
                    setupCommand();
                } else {
                    // User did not enable Bluetooth or an error occurred
                    Toast.makeText(this, R.string.bt_not_enabled_leaving, Toast.LENGTH_SHORT).show();
                    finish();
                }
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.option_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.scan:
                // Launch the DeviceListActivity to see devices and do scan
                Intent serverIntent = new Intent(this, DeviceListActivity.class);
                startActivityForResult(serverIntent, REQUEST_CONNECT_DEVICE);
                return true;
            case R.id.discoverable:
                // Ensure this device is discoverable by others
                ensureDiscoverable();
                return true;
        }
        return false;
    }


    private void handleMessage(Message msg) {
        switch (msg.what) {
            case MESSAGE_DEVICE_NAME:
                // save the connected device's name
                String mConnectedDeviceName = msg.getData().getString(DEVICE_NAME);
                Toast.makeText(getApplicationContext(), R.string.connected_to
                        + mConnectedDeviceName, Toast.LENGTH_SHORT).show();
                break;
            case MESSAGE_CONNECT_FAILED:
                Toast.makeText(getApplicationContext(), R.string.unable_to_connect,
                        Toast.LENGTH_SHORT).show();
                break;
            case MESSAGE_CONNECT_LOST:
                Toast.makeText(getApplicationContext(), R.string.connection_lost,
                        Toast.LENGTH_SHORT).show();
                break;
        }
    }


    private static class IncomingHandler extends Handler {
        private final WeakReference<MainActivity> mActivity;

        IncomingHandler(MainActivity activity) {
            mActivity = new WeakReference<>(activity);
        }

        @Override
        public void handleMessage(Message msg) {
            MainActivity activity = mActivity.get();
            if (activity != null) {
                activity.handleMessage(msg);
            }
        }
    }

}